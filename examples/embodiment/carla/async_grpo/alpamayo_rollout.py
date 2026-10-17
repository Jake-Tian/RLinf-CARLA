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

"""Optionally pace rollout with RLinf's existing staleness gate.

Set RLINF_PACE_TO_STALENESS=1 to avoid producing epochs the actor cannot use.
Pacing is off by default and does not change the sync/async config comparison.
"""

from __future__ import annotations

import asyncio
import os

from omegaconf.omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


def pace_requested() -> bool:
    """Whether the rollout should wait out the staleness window before each epoch."""
    return os.environ.get("RLINF_PACE_TO_STALENESS", "0") == "1"


class PacedAsyncMultiStepRolloutWorker(AsyncMultiStepRolloutWorker):
    """``AsyncMultiStepRolloutWorker`` that holds the staleness window per epoch."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.pace_to_staleness = pace_requested()
        self._full_weight_sync_lock = (
            asyncio.Lock()
            if cfg.actor.model.get("full_parameter_training", False)
            and cfg.weight_syncer.type == "bucket"
            else None
        )

    async def sync_model_from_actor(self):
        if self._full_weight_sync_lock is None:
            return await super().sync_model_from_actor()
        async with self._full_weight_sync_lock:
            return await super().sync_model_from_actor()

    async def _predict_rollout_actions_with_sync_guard(self, *args, **kwargs):
        if self.cfg.rollout.get("yield_before_prediction", False):
            await asyncio.sleep(0)
        return await super()._predict_rollout_actions_with_sync_guard(*args, **kwargs)

    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        # finished_episodes is None until the first weight sync lands, and wait_if_stale
        # asserts on it, so the epochs before that one go unpaced.
        def record(event):
            if os.environ.get("RLINF_PACING_AUDIT") != "1":
                return
            import json
            import time
            from pathlib import Path

            path = (
                Path(str(self.cfg.runner.logger.log_path))
                / f"pacing_rank_{self._rank}.jsonl"
            )
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": event,
                            "time": time.time(),
                            "rank": int(self._rank),
                            "version": int(self.version),
                            "finished_episodes": self.finished_episodes,
                            "envs_per_rank": self.total_num_train_envs,
                            "rollout_epoch": self.rollout_epoch,
                            "sync_in_flight": self._weight_sync_work is not None,
                        }
                    )
                    + "\n"
                )

        record("before_wait")
        if self.pace_to_staleness and self.finished_episodes is not None:
            with self.worker_timer("rollout/staleness_wait"):
                await self.wait_if_stale()
        record("after_wait")
        result = await super().generate_one_epoch(input_channel, output_channel)
        record("epoch_complete_before_counter_increment")
        return result
