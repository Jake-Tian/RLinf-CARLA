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

"""Paced async rollout: RLinf's async rollout worker that does not run ahead.

``AsyncMultiStepRolloutWorker._generate`` calls ``generate_one_epoch`` in a bare
``while True``, and ``PriorityStore.add`` discards the surplus rather than blocking
the producer, so the rollout keeps producing epochs the actor will never train on --
about 4.4 wasted on 4-7 for every 1 consumed. Those cards hold the actor, and 14431
spent ``actor/run_training`` = 452.9 s per step against the sync arm's 281.4 s for the
same one epoch of useful data per step. The waste is not free: it is taken off the
actor's own compute.

The gate is RLinf's own ``wait_if_stale`` -- the call ``decoupled_generate_one_epoch``
already makes for the decoupled path, so this adds no new mechanism. It is armed by
``RLINF_PACE_TO_STALENESS`` rather than a config key because the sync and async
configs may differ only in their two declared knobs (tests/test_async_config_ab.py
asserts that line by line), and it is off by default, leaving the async arm's
existing semantics untouched.

What this does and does not fix: it removes the wasted epochs, so the actor stops
sharing 4-7 with work nobody will use. Whether that is enough to reach the sync arm's
281 s depends on ``wait_if_stale``'s lookahead, which is ``staleness_threshold +
version`` countable in produced epochs and is therefore read from
``construct_rollout_batch`` in the run, not argued here.
"""

from __future__ import annotations

import os

from omegaconf.omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
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

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        # finished_episodes is None until the first weight sync lands, and wait_if_stale
        # asserts on it, so the epochs before that one go unpaced.
        if self.pace_to_staleness and self.finished_episodes is not None:
            await self.wait_if_stale()
        return await super().generate_one_epoch(input_channel, output_channel)
