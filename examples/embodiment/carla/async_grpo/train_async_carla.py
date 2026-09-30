# Copyright 2025 The RLinf Authors.
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
#
# Copied from RLinf's examples/embodiment/train_async.py so the async GRPO run
# differs from the async PPO run in one branch and nothing else. Submitted by
# The synchronous counterpart is train_embodied_agent.py.
# The rollout worker is RLinf's own subclassed in async_grpo/rollout.py, which is a
# no-op unless RLINF_PACE_TO_STALENESS=1, so 14431's arm is reproducible from here.

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

from .rollout import PacedAsyncMultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


# config_path=None on purpose: this file is not inside RLinf's examples/embodiment,
# so the decorator's relative "config" resolves nowhere. The sbatch passes the same
# --config-path/--config-name pair the synchronous entry script gets.
@hydra.main(
    version_base="1.1", config_path=None, config_name="carla_grpo_starvla_async"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
            AsyncEmbodiedSACFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "rlt_ac":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import AsyncRLTACFSDPPolicy

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncRLTACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
            AsyncEmbodiedDAGGERFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "decoupled_actor_critic":
        from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
        from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor

        runner_cls = AsyncPPOEmbodiedRunner
        actor_worker_cls = AsyncPPOEmbodiedFSDPActor
    elif cfg.algorithm.adv_type == "grpo":
        # Same runner as decoupled_actor_critic, which is already algorithm-agnostic:
        # it hands adv_type/group_size to calculate_adv_and_returns and only builds a
        # critic when loss_type asks for one. The actor subclass adds the group-layout
        # assertion GRPO needs; see async_grpo/actor.py.
        from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner

        from .actor import CarlaGRPOAsyncActor

        assert cfg.algorithm.loss_type == "actor", (
            f"adv_type=grpo computes advantages per group, which only pairs with the "
            f"loss registered as 'actor'; loss_type={cfg.algorithm.loss_type} would "
            f"feed grouped advantages to a different objective"
        )
        runner_cls = AsyncPPOEmbodiedRunner
        actor_worker_cls = CarlaGRPOAsyncActor
    else:
        raise ValueError(
            f"Unsupported loss type {cfg.algorithm.loss_type} for async embodied runner"
        )

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = PacedAsyncMultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = AsyncEnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        # Create reward worker group
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
