# Copyright 2026 The RLinf-CARLA Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Full-parameter sync actor with the async arm's audits and work counters."""

import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch

from examples.embodiment.carla.async_grpo.group_integrity import check_groups_intact
from examples.embodiment.carla.full_audit import audited_optimizer_step
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class FullColocateActor(EmbodiedFSDPActor):
    def init_worker(self):
        super().init_worker()
        from autopilot.models.alpamayo2 import sft_base_model
        from autopilot.utils.dataset import qwen_processor

        from rlinf.models.embodiment.alpamayo2 import policy

        sources = {
            "actor": inspect.getfile(EmbodiedFSDPActor),
            "policy": policy.__file__,
            "probe_actor": __file__,
            "model_compat": sft_base_model.__file__,
            "processor": qwen_processor.__file__,
        }
        record = {
            k: {
                "path": str(Path(v).resolve()),
                "sha256": hashlib.sha256(Path(v).read_bytes()).hexdigest(),
            }
            for k, v in sources.items()
        }
        (
            Path(str(self.cfg.runner.logger.log_path))
            / f"colocate_actor_rank{self._rank}_source.json"
        ).write_text(json.dumps(record, indent=2) + "\n")

    def compute_advantages_and_returns(self):
        t, b = self.rollout_batch["prev_logprobs"].shape[:2]
        v = self.rollout_batch.get("versions")
        bounds = None
        if v is not None:
            cols = v[:t].reshape(t, b, -1)
            bounds = [
                (float(cols[:, i].min()), float(cols[:, i].max())) for i in range(b)
            ]
        groups = check_groups_intact(
            b,
            bounds,
            int(self.cfg.algorithm.group_size),
            max_span=0,
            where="full colocate",
        )
        with (
            Path(str(self.cfg.runner.logger.log_path))
            / f"colocate_groups_rank{self._rank}.jsonl"
        ).open("a") as f:
            f.write(
                json.dumps(
                    {
                        "version": int(self.version),
                        "groups": groups,
                        "version_bounds": bounds,
                    }
                )
                + "\n"
            )
        return super().compute_advantages_and_returns()

    def run_training(self):
        t, b = self.rollout_batch["prev_logprobs"].shape[:2]
        samples = int(t * b)
        mask = self.rollout_batch.get("loss_mask")
        valid = (
            samples if mask is None else int(mask.reshape(samples, -1).any(-1).sum())
        )
        before = int(self.optimizer_steps)
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        result = super().run_training()
        actual = int(self.optimizer_steps) - before
        result.update(
            {
                "actor/actual_optimizer_steps": actual,
                "actor/actual_optimizer_steps_total": int(self.optimizer_steps),
                "actor/optimizer_steps_per_rank": actual,
                "actor/new_samples_per_rank": samples,
                "actor/valid_samples_per_rank": valid,
                "actor/valid_sample_fraction": valid / samples,
                "actor/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        row = {
            "version": int(self.version),
            "time_begin": started,
            "time_end": time.time(),
            "optimizer_before": before,
            "optimizer_after": int(self.optimizer_steps),
            "actual_global_updates": actual,
            "new_samples_per_rank": samples,
            "valid_samples_per_rank": valid,
        }
        with (
            Path(str(self.cfg.runner.logger.log_path))
            / f"colocate_update_rank{self._rank}.jsonl"
        ).open("a") as f:
            f.write(json.dumps(row) + "\n")
        return result

    def optimizer_step(self):
        if os.environ.get("RLINF_FULL_AUDIT") == "1":
            return audited_optimizer_step(self, super().optimizer_step)
        return super().optimizer_step()

    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        # Match the async arm's configured masking before the policy forward.
        def before(module, args, kwargs):
            if self.cfg.actor.get("skip_fully_masked_samples", False):
                mask = micro_batch.get("loss_mask")
                kwargs["loss_mask"] = (
                    mask.to(self.device) if torch.is_tensor(mask) else mask
                )
            return args, kwargs

        def after(module, args, kwargs, output):
            if self.optimizer_steps or os.environ.get("RLINF_FULL_AUDIT") != "1":
                return
            old = micro_batch["prev_logprobs"].to(output["logprobs"].device)
            delta = (output["logprobs"].detach().float() - old.float()).abs()
            index = getattr(self, "_full_first_micro_index", 0)
            self._full_first_micro_index = index + 1
            root = Path(str(self.cfg.runner.logger.log_path))
            with (root / f"initial_logprob_rank{self._rank}.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "max_abs": float(delta.max()),
                            "mean_abs": float(delta.mean()),
                            "microbatch": index,
                            "version": int(self.version),
                        }
                    )
                    + "\n"
                )
            if index == 0:

                def cpu(x):
                    if isinstance(x, dict):
                        return {k: cpu(v) for k, v in x.items()}
                    return x.detach().cpu() if torch.is_tensor(x) else x

                # Synchronous input need not contain recorded per-token versions.
                versions = micro_batch.get("versions", torch.empty(0))
                torch.save(
                    cpu(
                        {
                            "forward_inputs": kwargs["forward_inputs"],
                            "prev_logprobs": old,
                            "training_logprobs": output["logprobs"],
                            "loss_mask": micro_batch.get("loss_mask"),
                            "versions": versions,
                        }
                    ),
                    root / f"first_microbatch_rank{self._rank}.pt",
                )

        pre = self.model.register_forward_pre_hook(before, with_kwargs=True)
        post = (
            self.model.register_forward_hook(after, with_kwargs=True)
            if self.optimizer_steps == 0
            else None
        )
        try:
            return super().train_micro_batch(micro_batch, metrics, is_last=is_last)
        finally:
            pre.remove()
            if post is not None:
                post.remove()
