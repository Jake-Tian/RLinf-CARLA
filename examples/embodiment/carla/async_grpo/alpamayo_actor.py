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

"""Check async GRPO groups before advantage computation and training shuffle."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor

from .group_integrity import check_groups_intact, version_spread


class CarlaGRPOAsyncActor(AsyncPPOEmbodiedFSDPActor):
    """``AsyncPPOEmbodiedFSDPActor`` that refuses to train on a malformed group."""

    def init_worker(self):
        """Optionally use one captured LoRA initialization across allocations."""
        import hashlib
        import inspect

        from autopilot.models.alpamayo2 import sft_base_model
        from autopilot.utils.dataset import qwen_processor

        from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy
        from rlinf.hybrid_engines.fsdp.utils import apply_fsdp2_to_model
        from rlinf.models.embodiment.alpamayo2 import policy

        sources = {
            "framework_actor": inspect.getfile(AsyncPPOEmbodiedFSDPActor),
            "adapter_actor": __file__,
            "policy": policy.__file__,
            "model_compat": sft_base_model.__file__,
            "processor": qwen_processor.__file__,
            "fsdp_strategy": inspect.getfile(FSDP2Strategy),
            "fsdp_utils": inspect.getfile(apply_fsdp2_to_model),
        }
        (
            Path(str(self.cfg.runner.logger.log_path))
            / f"actor_rank_{self._rank}_source.json"
        ).write_text(
            json.dumps(
                {
                    name: {
                        "path": str(Path(value).resolve()),
                        "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest(),
                    }
                    for name, value in sources.items()
                },
                indent=2,
            )
        )
        import rlinf

        expected = (
            Path(rlinf.__file__).resolve().parent
            / "workers/actor/async_ppo_fsdp_worker.py"
        )
        assert Path(sources["framework_actor"]).resolve() == expected, sources
        super().init_worker()
        path = os.environ.get("RLINF_ACTOR_INITIAL_LORA")
        if not path:
            return
        from torch.distributed.tensor import distribute_tensor

        state = torch.load(path, map_location="cpu", weights_only=True)
        parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        assert parameters.keys() == state.keys(), "Captured LoRA parameter names differ"
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name].to(device=parameter.device, dtype=parameter.dtype)
                if hasattr(parameter, "device_mesh"):
                    value = distribute_tensor(
                        value, parameter.device_mesh, parameter.placements
                    )
                parameter.copy_(value)
        self.log_info(f"Loaded shared initial LoRA: {path}")

    def run_training(self) -> dict[str, float]:
        """Report actual optimizer work and memory alongside runner timing."""
        time_dim, batch_dim = self.rollout_batch["prev_logprobs"].shape[:2]
        samples = int(time_dim * batch_dim)
        mask = self.rollout_batch.get("loss_mask")
        valid = (
            samples if mask is None else int(mask.reshape(samples, -1).any(-1).sum())
        )
        _steps = samples // (self.cfg.actor.global_batch_size // self._world_size)
        capture = os.environ.get("RLINF_ACTOR_CAPTURE_DIR")
        if capture and not getattr(self, "_capture_started", False):
            self._capture_started = True
            Path(capture).mkdir(parents=True, exist_ok=True)
            state = self._full_trainable_tensors()
            if int(self._rank) == 0:
                torch.save(state, Path(capture) / "initial_lora.pt")
                (Path(capture) / "capture_config.json").write_text(
                    json.dumps(
                        {"max_episode_steps": int(self.cfg.env.train.max_episode_steps)}
                    )
                )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        optimizer_before = int(self.optimizer_steps)
        metrics = super().run_training()
        metrics["actor/actual_optimizer_steps"] = (
            int(self.optimizer_steps) - optimizer_before
        )
        metrics["actor/actual_optimizer_steps_total"] = int(self.optimizer_steps)
        elapsed = time.perf_counter() - started
        metrics.update(
            {
                "actor/optimizer_steps_per_rank": int(self.optimizer_steps)
                - optimizer_before,
                "actor/new_samples_per_rank": samples,
                "actor/valid_samples_per_rank": valid,
                "actor/valid_sample_fraction": valid / samples,
                "actor/train_samples_per_second_per_rank": samples / elapsed,
                "actor/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        if capture and int(self._rank) == 0:
            (Path(capture) / "reference_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
        return metrics

    def _full_trainable_tensors(self, gradients=False):
        state = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.grad if gradients else parameter.detach()
            if value is None:
                continue
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            state[name] = value.detach().cpu().clone()
        return state

    def optimizer_step(self):
        capture = os.environ.get("RLINF_ACTOR_CAPTURE_DIR")
        if capture and not getattr(self, "_capture_saved", False):
            self._capture_saved = True
            size = self.cfg.actor.global_batch_size // self._world_size

            def first_batch(value):
                if isinstance(value, dict):
                    return {key: first_batch(item) for key, item in value.items()}
                return (
                    value[:size].detach().cpu()
                    if isinstance(value, torch.Tensor)
                    else value
                )

            torch.save(
                first_batch(self.rollout_batch),
                Path(capture) / f"batch_rank_{self._rank}.pt",
            )
            gradients = self._full_trainable_tensors(gradients=True)
            if int(self._rank) == 0:
                scale = self.grad_scaler.get_scale()
                torch.save(
                    {key: value / scale for key, value in gradients.items()},
                    Path(capture) / "reference_gradients.pt",
                )
        if (
            self.cfg.actor.model.get("full_parameter_training", False)
            and os.environ.get("RLINF_FULL_AUDIT") == "1"
        ):
            from examples.embodiment.carla.full_audit import audited_optimizer_step

            return audited_optimizer_step(self, super().optimizer_step)
        return super().optimizer_step()

    async def construct_rollout_batch(self, max_trajectories: int | None = None):
        staleness_metrics = await super().construct_rollout_batch(max_trajectories)
        if os.environ.get("RLINF_ACTOR_CONSUME_ONCE", "0") == "1":
            # topn marks items used but retains them. This experiment fixes reuse
            # at one rollout batch, with update_epoch controlling optimizer reuse.
            selected = list(self.rollout_store.sl[-self.rollout_store_size :])
            for item in selected:
                self.rollout_store.sl.remove(item)
                self.rollout_store._used_seqs.discard(item[1])
            staleness_metrics["consume_once/groups"] = len(selected)

        # [n_chunk, rollout_epoch x bsz, num_action_chunks], and rollout_epoch is 1
        # here, so axis 1 is the bsz that calculate_scores reshapes by group_size.
        bsz = int(self.rollout_batch["rewards"].shape[1])
        group_size = int(self.cfg.algorithm.group_size)
        max_span = int(self.cfg.algorithm.get("staleness_threshold", 0) or 0)
        bounds = self._version_bounds(bsz)
        n_groups = check_groups_intact(
            bsz, bounds, group_size, max_span=max_span, where="async rollout batch"
        )
        if bounds is not None:
            spread = version_spread(bounds, group_size)
            staleness_metrics.update({f"version/{k}": v for k, v in spread.items()})
            self._write_version_audit(n_groups, bsz, group_size, spread)
        self.log_info(
            f"grpo group layout ok: bsz={bsz} group_size={group_size} "
            f"groups={n_groups} max_span={max_span}"
        )
        staleness_metrics["train/grpo_groups"] = n_groups
        return staleness_metrics

    def _version_bounds(self, bsz: int) -> list[tuple[float, float]] | None:
        """Return per-trajectory version bounds, including weight syncs mid-flight."""
        versions = self.rollout_batch.get("versions")
        if versions is None:
            self.log_info("no versions in rollout batch; version window unchecked")
            return None
        assert versions.dim() >= 2 and int(versions.shape[1]) == bsz, (
            f"versions shape {tuple(versions.shape)} does not put the batch axis "
            f"(length {bsz}) second, so the per-group version check would read the "
            f"wrong axis and pass vacuously. Expected [n_chunk, {bsz}, ...] like rewards"
        )
        cols = versions.reshape(int(versions.shape[0]), bsz, -1)
        lo = cols.amin(dim=(0, 2)).tolist()
        hi = cols.amax(dim=(0, 2)).tolist()
        return list(zip(lo, hi))

    def _write_legacy_version_audit(self, n_groups, bsz, group_size, spread) -> None:
        """Write nonfatal diagnostics outside Ray worker stdout and metrics."""
        if int(self._rank) != 0:
            return
        path = os.path.join(str(self.cfg.runner.logger.log_path), "version_audit.jsonl")
        row = {
            "version": int(self.version),
            "bsz": bsz,
            "group_size": group_size,
            "n_groups": n_groups,
            **spread,
        }
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError as exc:
            self.log_info(f"version audit not written to {path}: {exc}")

    async def sync_model_to_rollout(self) -> None:
        version = int(
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        begin = int(self.optimizer_steps)
        await super().sync_model_to_rollout()
        end = int(self.optimizer_steps)
        if not hasattr(self, "_published_optimizer_ranges"):
            self._published_optimizer_ranges = {}
        self._published_optimizer_ranges[version] = (begin, end)
        if int(self._rank) == 0 or os.environ.get("RLINF_ALL_RANK_AGE_AUDIT") == "1":
            path = os.path.join(
                str(self.cfg.runner.logger.log_path),
                (
                    "publication_optimizer_audit.jsonl"
                    if int(self._rank) == 0
                    else f"publication_optimizer_audit_rank{self._rank}.jsonl"
                ),
            )
            with open(path, "a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "version": version,
                            "optimizer_steps_at_request": begin,
                            "optimizer_steps_at_completion": end,
                            "stable_during_sync": begin == end,
                        }
                    )
                    + "\n"
                )

    def _write_version_audit(self, n_groups, bsz, group_size, spread) -> None:
        self._write_legacy_version_audit(n_groups, bsz, group_size, spread)
        if int(self._rank) != 0 and os.environ.get("RLINF_ALL_RANK_AGE_AUDIT") != "1":
            return
        versions = self.rollout_batch["versions"].long()
        current = int(self.optimizer_steps)
        updates = (
            versions.shape[0]
            * bsz
            // (self.cfg.actor.global_batch_size // self._world_size)
        )
        mapping = getattr(self, "_published_optimizer_ranges", {})
        ages_low, ages_high, unknown = [], [], []
        for version in versions.unique().tolist():
            if int(version) not in mapping:
                unknown.append(int(version))
                continue
            begin, end = mapping[int(version)]
            count = int((versions == version).sum())
            ages_low.extend([current - end] * count)
            ages_high.extend([current - begin] * count)
        row = {
            "actor_version": int(self.version),
            "optimizer_steps_before_training": current,
            "optimizer_updates_in_batch": int(updates),
            "unknown_behavior_versions": unknown,
            "published_optimizer_ranges": mapping,
            "scope": "bounds from requested/completed publication steps; no training/filter change",
        }
        if ages_low:
            row.update(
                {
                    "age_before_min_lower": min(ages_low),
                    "age_before_max_upper": max(ages_high),
                    "age_before_mean_lower": sum(ages_low) / len(ages_low),
                    "age_before_mean_upper": sum(ages_high) / len(ages_high),
                    "age_during_training_max_upper_bound": max(ages_high)
                    + int(updates)
                    - 1,
                }
            )
        path = os.path.join(
            str(self.cfg.runner.logger.log_path),
            (
                "optimizer_age_audit.jsonl"
                if int(self._rank) == 0
                else f"optimizer_age_audit_rank{self._rank}.jsonl"
            ),
        )
        with open(path, "a") as stream:
            stream.write(json.dumps(row) + "\n")
