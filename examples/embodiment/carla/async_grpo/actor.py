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

"""Async GRPO actor: RLinf's async PPO actor, plus the one check GRPO needs.

The parent is algorithm-agnostic. It forwards ``adv_type``/``group_size`` to
``calculate_adv_and_returns``, never touches a value head, and skips
``compute_values`` unless ``adv_type == "gae"`` -- so ``grpo`` rides the identical
code path and only the advantages differ. What GRPO adds is that those advantages
read the batch axis *in groups* (``calculate_scores`` reshapes it as
``(-1, group_size)``), a layout PPO's per-sample advantages do not care about. So
exactly one method is overridden, to assert that layout before anything reads it.

Placement matters. ``run_training`` shuffles via ``flatten_rollout_batch_for_train``,
which destroys the grouping outright; that is safe only because the runner computes
advantages first. The assertion therefore has to run at construction, before both.
"""

from __future__ import annotations

import json
import os

from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor

from .group_integrity import check_groups_intact, version_spread


class CarlaGRPOAsyncActor(AsyncPPOEmbodiedFSDPActor):
    """``AsyncPPOEmbodiedFSDPActor`` that refuses to train on a malformed group."""

    async def construct_rollout_batch(self, max_trajectories: int | None = None):
        staleness_metrics = await super().construct_rollout_batch(max_trajectories)

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
        """Per-trajectory (min, max) weight version, or None if unavailable.

        ``versions`` is ``full_like(prev_logprobs)`` written once per generated
        chunk and concatenated along the trajectory, so its range over a column is
        the span of policies that produced that trajectory. A span of one version
        is what ``staleness_threshold: 1`` buys, not evidence of a fault -- the
        actor admits a trajectory on ``versions.min()``.

        A span needs no version term in the loss: ``prev_logprobs`` is written by
        the policy that generated each chunk, so it is already that chunk's
        behaviour logprob and the ratio is correct token-by-token.
        ``compute_ppo_actor_loss`` reads neither ``versions`` nor
        ``proximal_logprobs`` -- both are swallowed by ``**kwargs``.
        """
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

    def _write_version_audit(self, n_groups, bsz, group_size, spread) -> None:
        """Append one JSON line to the run dir. Diagnostics must never be fatal.

        A file rather than metrics: the actor's returned dict does not reach
        metrics.log in this RLinf build, and a Ray worker's stdout does not reach
        run.log, so this is the only copy Phase 3 can read back.
        """
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
