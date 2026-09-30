#!/usr/bin/env python3
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

"""Check RLinf can consume the CARLA env config: a typo'd key is silently a runtime default."""

from __future__ import annotations

import os
import re
import sys
import warnings

# Invoking a script by path puts its own directory on sys.path, not the cwd.
sys.path.insert(0, os.getcwd())

CONFIG_DIR = "examples/embodiment/config"
ENV_YAML = os.path.join(CONFIG_DIR, "env", "carla.yaml")
EXPERIMENT_YAML = os.path.join(CONFIG_DIR, "carla_grpo_starvla.yaml")
ENV_SRC = "rlinf/envs/sim/carla/env.py"

#: Keys CarlaEnv reads, listed by hand because a scraper would agree with itself.
REQUIRED = [
    "auto_reset",
    "ignore_terminations",
    "is_eval",
    "use_fixed_reset_state_ids",
    "max_episode_steps",
    "history_len",
    "server_dir",
    "base_port",
    "quality_level",
    "startup_timeout",
    "reward_mode",
    "time_cost",
    "fps",
    "image_width",
    "image_height",
    "action_mode",
    "max_lateral_offset_m",
    "route_file",
    "route_length_m",
]


def _resolve(node, root):
    """Resolve Hydra-style ``${a.b.c}`` refs; an unresolvable one is returned unchanged."""
    if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
        if node.startswith("${oc.env:"):
            return os.environ.get(node[9:-1], node)
        cur = root
        for part in node[2:-1].split("."):
            if not isinstance(cur, dict) or part not in cur:
                return node
            cur = cur[part]
        return _resolve(cur, root) if cur != node else node
    return node


def _world_size(placement) -> int:
    """'4-7' -> 4, '3' -> 1. RLinf's own component_placement syntax."""
    if isinstance(placement, int):
        return 1
    text = str(placement).strip()
    if "-" in text:
        lo, hi = text.split("-", 1)
        return int(hi) - int(lo) + 1
    return len(text.split(","))


def _ranks(placement) -> list:
    """'4-7' -> [4, 5, 6, 7], '0,2' -> [0, 2]. The accelerator ids themselves."""
    if placement is None:
        return []
    if isinstance(placement, int):
        return [placement]
    out = []
    for part in str(placement).strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


#: Every accelerator on the node; UE's -graphicsadapter picks any index, unlike DRI_PRIME.
REACHABLE_CARDS = tuple(range(8))
#: Above 4 servers per card is fatal, and 4 is the edge rather than a safe count.
MAX_SERVERS_PER_CARD = 4


def merge_env_split(env_yaml, exp, split):
    """The env keys for one split: carla.yaml is the Hydra base, the experiment's env.<split> the override."""
    merged = dict(env_yaml or {})
    merged.update((exp.get("env") or {}).get(split) or {})
    return merged


#: Keys starVLA's action-head builder reads with NO default; a missing one raises.
STARVLA_WINDOW_KEYS = ("future_action_window_size", "past_action_window_size")


def check_starvla_action_windows(action_model, failures, source=""):
    """The checkpoint's OWN config.yaml must carry the action-window keys.

    It is not the launch config: config_tracker.py dumps only *accessed* keys.
    A wrong window still passes load_state_dict(strict=True); it serves garbage quietly.
    """
    where = f" ({source})" if source else ""
    if not isinstance(action_model, dict):
        failures.append(
            f"the checkpoint's framework.action_model{where} is missing or not a "
            f"mapping, so starVLA cannot build the action head at all"
        )
        return
    missing = [k for k in STARVLA_WINDOW_KEYS if k not in action_model]
    if missing:
        failures.append(
            f"the checkpoint's config.yaml is missing {missing} under "
            f"framework.action_model{where}. starVLA reads these with no default "
            f"and raises ConfigAttributeError at model construction. They are "
            f"absent because config_tracker.py dumps only *accessed* keys, and "
            f"SFT's data path reads action_horizon instead. Run "
            f"repair_starvla_ckpt_config.py --run-dir <RUN_DIR>; the values are "
            f"in that dir's config.full.yaml."
        )
        return
    past = int(action_model[STARVLA_WINDOW_KEYS[1]])
    future = int(action_model[STARVLA_WINDOW_KEYS[0]])
    horizon = action_model.get("action_horizon")
    if horizon is not None and past + 1 + future != int(horizon):
        failures.append(
            f"the checkpoint's config.yaml says past={past} future={future} but "
            f"action_horizon={horizon}{where}; past+1+future = {past + 1 + future}. "
            f"The head would be built with the wrong chunk_len and, because "
            f"chunk_len is only a reshape, would load without complaint."
        )
    else:
        print(
            f"checkpoint action windows{where}: past={past} future={future} "
            f"-> chunk_len={past + 1 + future}"
        )


def checkpoint_run_dir(model_path):
    """The run dir for a starVLA checkpoint path, where config.yaml and dataset_statistics.json live."""
    path = os.path.abspath(str(model_path))
    if path.endswith(".pt"):
        return os.path.dirname(os.path.dirname(path))
    return path


def read_checkpoint_action_model(run_dir):
    """``framework.action_model`` from a run dir's config.yaml, via a parser independent of the repair script."""
    import yaml

    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return ((cfg or {}).get("framework") or {}).get("action_model")


def check_experiment_dict(exp, failures, verbose=True):
    """The arithmetic itself, over an already-parsed config; the local suite has no pyyaml."""

    def say(*args):
        if verbose:
            print(*args)

    def get(path, default=None):
        cur = exp
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return _resolve(cur, exp)

    placement = get("cluster.component_placement", {})
    env_world = _world_size(placement.get("env", 1))
    actor_world = _world_size(placement.get("actor", 1))
    stage_num = get("rollout.pipeline_stage_num", 1) or 1
    say(
        f"env_world_size={env_world} actor_world_size={actor_world} "
        f"pipeline_stage_num={stage_num}"
    )

    # The one silent failure here: too many servers per card kills them with SIGSEGV.
    env_ranks = _ranks(placement.get("env"))
    outside = sorted(set(env_ranks) - set(REACHABLE_CARDS))
    n_envs = get("env.train.total_num_envs") or 0
    per_proc = (n_envs // env_world) if env_world else 0
    if outside:
        failures.append(
            f"cluster.component_placement.env = {placement.get('env')!r} names "
            f"accelerator(s) {outside}, but this node has "
            f"{len(REACHABLE_CARDS)} cards {list(REACHABLE_CARDS)}."
        )
    elif per_proc > MAX_SERVERS_PER_CARD:
        total = len(env_ranks) * per_proc
        failures.append(
            f"cluster.component_placement.env = {placement.get('env')!r} with "
            f"env.train.total_num_envs {n_envs} starts {per_proc} CARLA servers "
            f"per env process, and every server of a process renders on that "
            f"process's card: {total} x ~5.87 GiB = {total * 5.87:.0f} GiB spread "
            f"over {len(env_ranks)} cards. 4 per card is the measured limit "
            f"(24.1 GiB) and the 5th exits rc=139 (probe 14397). Lower "
            f"total_num_envs, or lower env.train.group_size if the group can "
            f"shrink."
        )
    elif per_proc == MAX_SERVERS_PER_CARD:
        # Forced, not chosen: group_size >= 4 needs a whole group per env process.
        warnings.warn(
            f"env.train.total_num_envs {n_envs} over "
            f"{len(env_ranks)} env card(s) is {per_proc} CARLA servers per card, "
            f"which is the measured edge: 24.1 GiB on a 24 GiB card, and job "
            f"14400 lost one of eight servers at this density. group_size >= 4 "
            f"forces it, so expect it -- but a startup SIGSEGV here is this, not "
            f"a config typo.",
            stacklevel=2,
        )

    # An EQUALITY, not a divisibility: global_batch_size must equal the rollout's trajectory count.
    gbs = get("actor.global_batch_size")
    if gbs is not None and n_envs is not None and gbs != n_envs:
        failures.append(
            f"actor.global_batch_size {gbs} != env.train.total_num_envs "
            f"{n_envs}: embodied_fsdp_actor_worker.py:618-621 asserts the "
            f"trajectory count the rollout produced EQUALS global_batch_size"
        )

    # WANDB_MODE comes from the sbatch, so this checks the pair rather than the config
    # alone: wandb with no mode blocks instead of failing on a node with no route out.
    backends = get("runner.logger.logger_backends", []) or []
    if any("wandb" in str(b).lower() for b in backends):
        mode = os.environ.get("WANDB_MODE", "")
        if mode not in ("offline", "disabled"):
            failures.append(
                f"logger_backends lists wandb ({backends}) but WANDB_MODE is "
                f"{mode!r}: the GPU nodes have no internet route, so wandb.init() "
                f"blocks instead of erroring. Export WANDB_MODE=offline (runs land "
                f"in <log_path>/wandb/ and sync later), or drop wandb from the "
                f"backend list."
            )

    for split in ("train", "eval"):
        base = f"env.{split}"
        total = get(f"{base}.total_num_envs")
        group = get(f"{base}.group_size")
        horizon = get(f"{base}.max_steps_per_rollout_epoch")
        chunks = get("actor.model.num_action_chunks")
        if total is None or group is None:
            continue
        label = f"env.{split}"
        if total % env_world != 0:
            failures.append(
                f"{label}.total_num_envs {total} % env world size {env_world} != 0"
            )
        elif total // env_world // stage_num % group != 0:
            failures.append(
                f"{label}.total_num_envs {total} // {env_world} // {stage_num} "
                f"= {total // env_world // stage_num} is not divisible by "
                f"{label}.group_size {group} (so total_num_envs must be a "
                f"multiple of {env_world * stage_num * group})"
            )
        else:
            say(
                f"{label}: {total} envs / {group} = {total // group} groups, "
                f"{total // env_world // stage_num} per env process"
            )
        if horizon is not None and chunks and horizon % chunks != 0:
            failures.append(
                f"{label}.max_steps_per_rollout_epoch {horizon} % "
                f"actor.model.num_action_chunks {chunks} != 0"
            )

    micro = get("actor.micro_batch_size")
    gbs = get("actor.global_batch_size")
    if micro and gbs and gbs % (micro * actor_world) != 0:
        failures.append(
            f"actor.global_batch_size {gbs} % (micro_batch_size {micro} * "
            f"actor world size {actor_world}) != 0"
        )
    elif micro and gbs:
        say(
            f"actor: global_batch_size {gbs} / (micro {micro} * "
            f"world {actor_world}) = {gbs // (micro * actor_world)} mini-batches"
        )

    # Every actor rank must hold a WHOLE NUMBER OF GROUPS, since calculate_scores reshapes by group_size.
    alg_group = get("algorithm.group_size")
    if alg_group is None:
        alg_group = get("env.train.group_size")
    if gbs and alg_group and gbs % (actor_world * alg_group) != 0:
        failures.append(
            f"actor.global_batch_size {gbs} % (actor world size {actor_world} * "
            f"group_size {alg_group}) != 0: each actor rank gets "
            f"{gbs} // {actor_world} = {gbs // actor_world} trajectory(s), and "
            f"calculate_scores reshapes that by group_size "
            f"(algorithms/utils.py:143 after embodied_fsdp_actor_worker.py:603), "
            f"so every rank needs a multiple of {alg_group}. Raise the rank "
            f"count's divisor instead of global_batch_size: that number is "
            f"pinned equal to env.train.total_num_envs above."
        )
    elif gbs and alg_group:
        say(
            f"actor: {gbs // actor_world} trajectories per rank = "
            f"{gbs // (actor_world * alg_group)} groups per rank"
        )

    # A group's shared initial condition needs route_file to name ONE route, not a directory.
    route = get("env.train.route_file", None)
    if route is None:
        failures.append(
            "env.train.route_file is unset, so reset() builds the synthetic "
            "straight route -- route_completion can never reach the 100 that "
            "success requires, and every reward is a timeout"
        )
    elif str(route).endswith("/") or "*" in str(route):
        failures.append(
            f"env.train.route_file {route!r} looks like a directory; GRPO needs "
            f"one shared route, not a per-reset sample"
        )

    # Train must not auto-reset or ignore terminations, or RLinf never builds loss_mask.
    train_auto_reset = get("env.train.auto_reset", None)
    train_ignore = get("env.train.ignore_terminations", False)
    if train_auto_reset:
        failures.append(
            "env.train.auto_reset is truthy, so RLinf never builds loss_mask "
            "and compute_grpo_advantages raises TypeError on "
            "torch.zeros_like(None); set it False for train"
        )
    if train_ignore:
        failures.append(
            "env.train.ignore_terminations is truthy, which likewise suppresses "
            "loss_mask and zeroes every termination the advantage depends on"
        )

    # A WARNING, not a failure: LoRA is forced by arithmetic and that wrapper path is
    # unproven, so the smoke stage's reward is what tests it.
    if get("actor.model.is_lora", False):
        # warnings.warn, not print: it reaches stderr under sbatch and stays findable.
        warnings.warn(
            "actor.model.is_lora is True. The SFT checkpoint is a merged model "
            "with no lora_A/lora_B, so this relies on a fresh zero-init LoRA "
            "pair (delta 0, policy == the merged checkpoint) on a wrapper path "
            "that openvla/openvlaoft exercises but StarVLA does not. Full "
            "finetune did not fit alongside CARLA on the 2 actor cards it was "
            "measured on, and that count is unverified at a different one, so this is "
            "forced -- but check the smoke stage's reward before trusting the "
            "run.",
            stacklevel=2,
        )

    # A Qwen2.5 class name matches nothing in a Qwen3-VL model; no assert catches it.
    wrapped = (
        get("actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap", []) or []
    )
    stale = [w for w in wrapped if "Qwen2_5" in str(w) or "Qwen2." in str(w)]
    if stale:
        failures.append(
            f"wrap_policy names Qwen2.5 classes {stale}, which match nothing in "
            f"a Qwen3-VL model; the wrap silently degrades to the root module"
        )
    elif wrapped:
        say(f"wrap_policy: {wrapped}")


def check_checkpoint_config(exp, failures, ckpt_reader=None, isdir=os.path.isdir):
    """Resolve the checkpoint from the composed config, then check the config.yaml it builds from."""
    ckpt_reader = ckpt_reader or read_checkpoint_action_model
    model_path = exp.get("actor", {}).get("model", {}).get("model_path")
    if model_path is None:
        model_path = exp.get("rollout", {}).get("model", {}).get("model_path")
    model_path = _resolve(model_path, exp)
    if not model_path:
        failures.append(
            "neither actor.model.model_path nor rollout.model.model_path is set, "
            "so there is no checkpoint to load"
        )
        return
    run_dir = checkpoint_run_dir(model_path)
    if not isdir(run_dir):
        failures.append(
            f"checkpoint run dir {run_dir} does not exist (from model_path "
            f"{model_path}); its config.yaml is what the action head is built from"
        )
        return
    try:
        action_model = ckpt_reader(run_dir)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"cannot read {run_dir}/config.yaml: {exc}")
        return
    check_starvla_action_windows(action_model, failures, source=run_dir)


def check_experiment_config(failures, path=None, ckpt_reader=None, isdir=os.path.isdir):
    """Check the cross-key arithmetic here; RLinf's own asserts then report as a sentence."""
    path = path or EXPERIMENT_YAML
    try:
        import yaml

        with open(path) as f:
            exp = yaml.safe_load(f)
        with open(os.path.join(os.path.dirname(path), "env", "carla.yaml")) as f:
            env_yaml = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"cannot parse {path}: {exc}")
        return
    print(f"parsed {path}")
    exp.setdefault("env", {})
    for split in ("train", "eval"):
        exp["env"][split] = merge_env_split(env_yaml, exp, split)
    check_experiment_dict(exp, failures)

    print()
    check_checkpoint_config(exp, failures, ckpt_reader=ckpt_reader, isdir=isdir)


def _resolve_experiment_yaml(name_or_path: str | None) -> str:
    """A bare name resolves under CONFIG_DIR; anything with a separator is used as given.

    The sbatch passes its own config name, so the async variant is never
    silently checked against the sync file.
    """
    if not name_or_path:
        return EXPERIMENT_YAML
    if os.sep in name_or_path:
        return name_or_path
    base = name_or_path if name_or_path.endswith(".yaml") else name_or_path + ".yaml"
    return os.path.join(CONFIG_DIR, base)


def main(experiment_yaml=None) -> int:
    failures = []

    # 1. Plain parse first, so a malformed yaml is not reported as a Hydra error.
    try:
        import yaml

        with open(ENV_YAML) as f:
            raw = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot parse {ENV_YAML}: {exc}")
        return 1
    print(f"parsed {ENV_YAML}: {len(raw)} keys")

    # 2. Every key the env reads must be in the file.
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        failures.append(f"yaml is missing keys CarlaEnv reads: {missing}")
    extra = [
        k
        for k in raw
        if k not in REQUIRED
        and k
        not in (
            "env_type",
            "total_num_envs",
            "max_steps_per_rollout_epoch",
            "use_ordered_reset_state_ids",
            "use_rel_reward",
            "reward_coef",
            "seed",
            "group_size",
            "video_cfg",
            "gpu",
        )
    ]
    if extra:
        # Not fatal, but an unrecognised key is usually a typo taking a silent default.
        print(f"note: keys CarlaEnv never reads (typo?): {extra}")

    # 3. env_type must resolve through RLinf's own enum and dispatch.
    try:
        from rlinf.envs import get_env_cls

        cls = get_env_cls(raw["env_type"])
        print(f"get_env_cls({raw['env_type']!r}) -> {cls.__name__}")
        if cls.__name__ != "CarlaEnv":
            failures.append(f"dispatch returned {cls.__name__}, not CarlaEnv")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"RLinf dispatch failed: {exc}")

    # 4. Cross-check against env.py's .get() defaults: a misspelled key that is defaulted hides above.
    try:
        with open(ENV_SRC) as f:
            src = f.read()
        # Anchored on self.cfg: a bare .get( also matches self._outcome_tally.get("success").
        asked = set(re.findall(r"(?:self\.)?\bcfg\.get\(\s*[\"']([a-z_]+)[\"']", src))
        asked |= set(re.findall(r"self\.cfg\.([a-z_]+)\b(?!\s*\()", src))
        unread = sorted(asked - set(raw))
        if unread:
            failures.append(
                f"env.py reads keys absent from the yaml (they will silently "
                f"take defaults): {unread}"
            )
        else:
            print(f"all {len(asked)} keys env.py reads are present in the yaml")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"could not cross-check against {ENV_SRC}: {exc}")

    # 5. The experiment config's cross-key arithmetic, as RLinf's validator asserts it.
    print()
    check_experiment_config(failures, path=_resolve_experiment_yaml(experiment_yaml))

    if failures:
        print()
        for f_ in failures:
            print(f"FAIL: {f_}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    # Optional: a config name or a path; defaults to the sync experiment.
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
