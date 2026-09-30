# Copyright 2026 The RLinf-CARLA Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CarlaEnv: the RLinf vector-env contract (reset/step/chunk_step) that RLinf calls."""

from __future__ import annotations

import copy
import dataclasses
from collections import deque
from typing import Optional, Union

import numpy as np
import torch

from .reward import RewardConfig
from .server import ServerConfig
from .worker import CarlaEnvWorker, CarlaWorkerConfig


def resolve_worker_gpu(worker_info, worker_index: int, total_num_processes: int) -> int:
    """Which GPU a CARLA server should render on."""
    if isinstance(worker_info, dict):
        gpu = worker_info.get("gpu", None)
    else:
        gpu = getattr(worker_info, "accelerator_rank", None)

    # -1 is what RLinf writes when a worker was placed on no accelerator.
    if gpu is None or int(gpu) < 0:
        # Only correct single-process, where no other process can disagree.
        gpu = worker_index % max(1, total_num_processes)
    return int(gpu)


def server_port_index(seed_offset: int, num_envs: int, worker_index: int) -> int:
    """A CARLA server's index among all servers this job starts; worker_index is process-local."""
    return seed_offset * num_envs + worker_index


class CarlaEnv:
    """A vector of CARLA driving envs behind RLinf's env interface."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
        logdir: str = "carla_env_logs",
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.logdir = logdir

        self.auto_reset = bool(cfg.get("auto_reset", True))
        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self.is_eval = bool(cfg.get("is_eval", False))
        self.use_fixed_reset_state_ids = bool(
            cfg.get("use_fixed_reset_state_ids", False)
        )
        self.max_episode_steps = int(cfg.get("max_episode_steps", 1200))

        #: Past frames exposed as frame_history; main_images must stay single-frame.
        self.history_len = int(cfg.get("history_len", 1))

        self.workers: list[CarlaEnvWorker] = []
        self._history: list[deque] = []
        #: Raw per-env obs from the last step(); re-reading the worker drains its queue.
        self._last_raw_obs: Optional[list[dict]] = None
        self._init_workers()

        self.task_descriptions = ["drive"] * self.num_envs
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._is_start = True
        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._init_metrics()

    # -- construction ------------------------------------------------------

    def _init_workers(self) -> None:
        # `.get`, not attribute access: a plain-dict caller has no `cfg.server_dir`.
        server_dir = self.cfg.get("server_dir")
        if not server_dir:
            # Checked here so a missing key fails now, not as a server that never starts.
            raise ValueError(
                "CarlaEnv config has no 'server_dir'; cannot locate the CARLA "
                "server. Set it in the env config (env.train.server_dir)."
            )
        server_cfg = ServerConfig(
            server_dir=server_dir,
            base_port=int(self.cfg.get("base_port", 2100)),
            quality_level=self.cfg.get("quality_level", "Epic"),
            startup_timeout=float(self.cfg.get("startup_timeout", 240.0)),
        )
        reward_cfg = RewardConfig(
            mode=self.cfg.get("reward_mode", "sparse"),
            time_cost=self.cfg.get("time_cost", None),
        )
        for i in range(self.num_envs):
            worker_cfg = CarlaWorkerConfig(
                server=dataclasses.replace(
                    server_cfg,
                    gpu=resolve_worker_gpu(
                        self.worker_info, i, self.total_num_processes
                    ),
                    # Ports must be unique node-wide -- see server_port_index.
                    port_index=server_port_index(self.seed_offset, self.num_envs, i),
                ),
                reward=reward_cfg,
                fps=float(self.cfg.get("fps", 20.0)),
                image_width=int(self.cfg.get("image_width", 1280)),
                image_height=int(self.cfg.get("image_height", 720)),
                action_mode=self.cfg.get("action_mode", "control"),
                max_episode_steps=self.max_episode_steps,
                max_lateral_offset_m=float(self.cfg.get("max_lateral_offset_m", 2.0)),
                route_file=self.cfg.get("route_file", None),
                route_length_m=float(self.cfg.get("route_length_m", 200.0)),
                seed=self.seed_offset,
            )
            worker = CarlaEnvWorker(worker_cfg, i, self.logdir)
            worker.start()
            self.workers.append(worker)
            self._history.append(deque(maxlen=max(1, self.history_len)))

    def close(self) -> None:
        for w in self.workers:
            try:
                w.close()
            except Exception:  # noqa: BLE001 - teardown must not mask errors
                pass
        self.workers = []

    def __enter__(self) -> CarlaEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- properties RLinf reads -------------------------------------------

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    # -- metrics -----------------------------------------------------------

    def _init_metrics(self) -> None:
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)
        #: Per-episode outcome tally; decides whether the dense reward term is needed.
        self._outcome_tally: dict[str, int] = {}

    def _reset_metrics(self, env_idx=None) -> None:
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0.0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        self.returns += step_reward * (~self.success_once)
        new_success = terminations & ~self.success_once
        if new_success.any():
            self.success_episode_len[new_success] = self._elapsed_steps[new_success]
        self.success_once = self.success_once | terminations

        episode_info = {
            "success_once": self.success_once.copy(),
            "return": self.returns.copy(),
            "episode_len": self._elapsed_steps.copy(),
        }
        lens = np.where(
            self.success_once, self.success_episode_len, self._elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(lens, 1)
        infos["episode"] = _to_tensor(episode_info)
        return infos

    def outcome_summary(self) -> dict:
        """Success/crash/timeout counts since the last reset, for reading after a rollout."""
        total = sum(self._outcome_tally.values())
        if total == 0:
            return {"total": 0}
        out = {"total": total}
        out.update(self._outcome_tally)
        out["success_rate"] = self._outcome_tally.get("success", 0) / total
        return out

    # -- observation -------------------------------------------------------

    def _wrap_obs(self, obs_list: list[dict], record_history: bool = False) -> dict:
        images = np.stack([o["main_images"] for o in obs_list])
        states = np.stack([o["states"] for o in obs_list])
        obs = {
            "main_images": _to_tensor(images),
            "states": _to_tensor(states),
            "task_descriptions": [o["task_descriptions"] for o in obs_list],
        }
        # wrist_images is absent on purpose; see CarlaEnvWorker._obs.

        if self.history_len > 1:
            # Opt-in: chunk_step would otherwise append the final frame twice.
            if record_history:
                for i, o in enumerate(obs_list):
                    self._history[i].append(o["main_images"])
            # Right-pad with the oldest frame: a ragged history would not stack.
            frames = []
            for h in self._history:
                pad = [h[0]] * (self.history_len - len(h)) if h else []
                frames.append(np.stack(list(pad) + list(h)))
            obs["frame_history"] = _to_tensor(np.stack(frames))
        return obs

    # -- episode control ---------------------------------------------------

    def update_reset_state_ids(self) -> None:
        """No-op, and required: RLinf calls it per rollout epoch outside any guard."""
        return None

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = set(np.asarray(env_idx).reshape(-1).tolist())

        # A partial reset still returns an obs for EVERY env; the rest keep their frame.
        obs_list = [None] * self.num_envs
        for idx in range(self.num_envs):
            if idx in env_idx:
                obs_list[idx] = self.workers[idx].reset()
                self._history[idx].clear()
            elif self._last_raw_obs is not None:
                obs_list[idx] = self._last_raw_obs[idx]
            else:
                raise RuntimeError(
                    f"cannot partially reset envs {sorted(env_idx)} before the "
                    f"first full reset: env {idx} has no observation to carry over"
                )

        self._reset_metrics(np.array(sorted(env_idx)))
        self._is_start = False
        self._last_raw_obs = obs_list
        return self._wrap_obs(obs_list, record_history=True), {}

    def step(self, actions=None, auto_reset: bool = True, _skip_obs_wrap: bool = False):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)

        self._elapsed_steps += 1
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for i, worker in enumerate(self.workers):
            o, r, term, trunc, info = worker.step(actions[i])
            obs_list.append(o)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
            self.task_descriptions[i] = o["task_descriptions"]

        step_reward = np.asarray(rewards, dtype=np.float32)
        terminations = np.asarray(terms, dtype=bool)
        truncations = np.asarray(truncs, dtype=bool)

        # Kept for chunk_step; _handle_auto_reset mutates it in place.
        self._last_raw_obs = obs_list

        if _skip_obs_wrap:
            obs = None
        else:
            obs = self._wrap_obs(obs_list, record_history=True)

        infos = _list_of_dict_to_dict_of_list(infos)
        infos = self._record_metrics(step_reward, terminations, infos)

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = _to_tensor(terminations)
            terminations = np.zeros_like(terminations)

        dones = terminations | truncations
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs_list, infos)

        return (
            obs,
            _to_tensor(step_reward),
            _to_tensor(terminations),
            _to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        """Step ``chunk_size`` ticks, wrapping only the final tick's observation."""
        chunk_size = chunk_actions.shape[1]
        obs_list, infos_list, chunk_rewards = [], [], []
        raw_terms, raw_truncs = [], []
        last_obs = None
        last_raw = None

        for i in range(chunk_size):
            should_render = i == chunk_size - 1
            obs, r, term, trunc, info = self.step(
                chunk_actions[:, i],
                auto_reset=False,
                _skip_obs_wrap=not should_render,
            )
            if should_render:
                # Reuse the final obs; a second read drains the queue and returns None.
                last_obs = obs
                last_raw = self._last_raw_obs
            obs_list.append(None)
            infos_list.append(info)
            chunk_rewards.append(r)
            raw_terms.append(term)
            raw_truncs.append(trunc)

        obs_list[-1] = last_obs

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_terms = torch.stack(raw_terms, dim=1)
        raw_truncs = torch.stack(raw_truncs, dim=1)

        past_terms = raw_terms.any(dim=1)
        past_truncs = raw_truncs.any(dim=1)
        past_dones = torch.logical_or(past_terms, past_truncs)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), last_raw, infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terms = torch.zeros_like(raw_terms)
            chunk_terms[:, -1] = past_terms
            chunk_truncs = torch.zeros_like(raw_truncs)
            chunk_truncs[:, -1] = past_truncs
        else:
            chunk_terms = raw_terms.clone()
            chunk_truncs = raw_truncs.clone()

        return obs_list, chunk_rewards, chunk_terms, chunk_truncs, infos_list

    def _handle_auto_reset(self, dones, obs_list, infos):
        """Reset the finished envs and re-wrap the vector, keeping the other frames."""
        dones = np.asarray(dones, dtype=bool)
        env_idx = np.arange(self.num_envs)[dones]
        final_obs = list(obs_list) if obs_list is not None else None
        final_info = copy.deepcopy(infos)

        for idx in env_idx:
            obs_list[idx] = self.workers[idx].reset()
            self._history[idx].clear()
        self._reset_metrics(env_idx)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return self._wrap_obs(obs_list, record_history=True), infos


# -- helpers ---------------------------------------------------------------


def _to_tensor(value):
    """Batch to torch, matching rlinf.envs.utils.to_tensor's contract."""
    if isinstance(value, dict):
        return {k: _to_tensor(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def _list_of_dict_to_dict_of_list(items):
    if not items:
        return {}
    out = {k: [] for k in items[0]}
    for item in items:
        for k, v in item.items():
            out[k].append(v)
    return out
