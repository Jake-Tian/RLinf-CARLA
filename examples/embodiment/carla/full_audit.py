# Copyright 2026 The RLinf-CARLA Contributors.
# SPDX-License-Identifier: Apache-2.0

"""First-update evidence for the full-parameter experiment, outside timing."""

import json
from pathlib import Path

import torch


def role(name):
    if ".visual." in name:
        return "vision"
    if ".language_model." in name:
        return "language"
    return "projection_or_other"


def local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def gradient_energy(actor):
    energies = {
        key: torch.zeros((), device=actor.device)
        for key in ("vision", "language", "projection_or_other")
    }
    samples = {}
    missing = []
    for name, parameter in actor.model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        grad = local(parameter.grad.detach()).float()
        key = role(name)
        energies[key] += grad.square().sum()
        prefix = grad.reshape(-1)[:64]
        if key not in samples and prefix.numel() and bool(prefix.abs().max() > 0):
            samples[key] = (
                name,
                local(parameter.detach()).reshape(-1)[:64].cpu().clone(),
            )
    values = torch.stack(list(energies.values()))
    torch.distributed.all_reduce(values)
    assert bool(torch.isfinite(values).all()), values
    return {key: float(value) for key, value in zip(energies, values)}, samples, missing


def audited_optimizer_step(actor, step):
    if getattr(actor, "_full_first_update_audited", False):
        return step()
    before, samples, missing = gradient_energy(actor)
    norm, lr = step()
    after, _, _ = gradient_energy(actor)
    parameters = dict(actor.model.named_parameters())
    changes = {
        key: {
            "name": name,
            "max_abs_delta": float(
                (local(parameters[name].detach()).reshape(-1)[:64].cpu() - value)
                .abs()
                .max()
            ),
        }
        for key, (name, value) in samples.items()
    }
    optimizer_dtypes = {}
    optimizer_steps = set()
    for state in actor.optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                dtype = str(state[key].dtype)
                optimizer_dtypes[dtype] = optimizer_dtypes.get(dtype, 0) + 1
        if "step" in state:
            optimizer_steps.add(float(state["step"]))
    full_before, full_after = sum(before.values()) ** 0.5, sum(after.values()) ** 0.5
    target = min(full_before, float(actor.cfg.actor.optim.clip_grad))
    record = {
        "rank": actor._rank,
        "full_preclip_norm": full_before,
        "reported_preclip_norm": float(norm),
        "full_postclip_norm": full_after,
        "gradient_energy_by_role": before,
        "sampled_parameter_changes": changes,
        "missing_gradients": missing,
        "optimizer_state_dtypes": optimizer_dtypes,
        "adam_steps": sorted(optimizer_steps),
        "run_optimizer_counter": int(actor.optimizer_steps),
        "lr": lr,
        "scope": "First actual update only. Global gradient norms plus representative local parameter changes and optimizer state dtypes; not full online learning equivalence.",
    }
    path = (
        Path(str(actor.cfg.runner.logger.log_path))
        / f"full_first_update_rank{actor._rank}.json"
    )
    path.write_text(json.dumps(record, indent=2) + "\n")
    assert abs(float(norm) / max(full_before, 1e-30) - 1) < 1e-5, record
    assert abs(full_after - target) < 0.002, record
    assert before["vision"] > 0 and before["language"] > 0, record
    assert set(optimizer_dtypes) == {"torch.float32"}, record
    actor._full_first_update_audited = True
    return norm, lr
