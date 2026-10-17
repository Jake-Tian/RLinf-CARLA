# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared trajectory-token fusion for model and data-pipeline callers."""

from __future__ import annotations

from typing import Any, Mapping

import einops
import torch
from hydra.utils import instantiate


def replace_placeholder_tokens(
    input_ids: torch.Tensor,
    replacement_ids: torch.Tensor,
    placeholder_id: int,
    *,
    component: str,
) -> torch.Tensor:
    """Replace one component's placeholders with explicit per-sample validation."""

    if input_ids.ndim != 2 or replacement_ids.ndim != 2:
        raise ValueError("input_ids and replacement_ids must both be rank-2 tensors")
    if input_ids.shape[0] != replacement_ids.shape[0]:
        raise ValueError(
            f"{component} batch mismatch: input batch={input_ids.shape[0]}, "
            f"replacement batch={replacement_ids.shape[0]}"
        )
    mask = input_ids == placeholder_id
    actual = mask.sum(dim=1)
    expected = replacement_ids.shape[1]
    mismatched = torch.where(actual != expected)[0]
    if mismatched.numel():
        details = ", ".join(
            f"sample {index.item()}: expected {expected}, found {actual[index].item()}"
            for index in mismatched[:8]
        )
        raise ValueError(f"{component} placeholder count mismatch ({details})")
    fused = input_ids.clone()
    return fused.masked_scatter(mask, replacement_ids.to(fused.device))


def tokenize_history_trajectory(
    tokenizer: Any,
    traj_data: Mapping[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    """Encode history with prefix shape ``[B, n_traj, ...]``."""

    history_xyz = traj_data.get("ego_history_xyz")
    history_rot = traj_data.get("ego_history_rot")
    if history_xyz is None or history_rot is None:
        raise ValueError("history trajectory requires ego_history_xyz and ego_history_rot")
    if history_xyz.ndim != 4:
        raise ValueError("ego_history_xyz must have shape [B, n_traj, T, 3]")
    if history_rot.ndim != 5:
        raise ValueError("ego_history_rot must have shape [B, n_traj, T, 3, 3]")

    batch_size = history_xyz.shape[0]
    flat_xyz = history_xyz.flatten(start_dim=0, end_dim=1)
    flat_rot = history_rot.flatten(start_dim=0, end_dim=1)
    history_ids = tokenizer.encode(
        hist_xyz=flat_xyz[:, :1],
        hist_rot=flat_rot[:, :1],
        fut_xyz=flat_xyz,
        fut_rot=flat_rot,
    )
    history_ids = history_ids + start_idx
    return einops.rearrange(history_ids, "(b n_traj) n -> b (n_traj n)", b=batch_size)


def tokenize_future_trajectory(
    tokenizer: Any,
    traj_data: Mapping[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    """Encode future trajectories with prefix shape ``[B, n_traj, ...]``."""

    future_xyz = traj_data.get("ego_future_xyz")
    future_rot = traj_data.get("ego_future_rot")
    history_xyz = traj_data.get("ego_history_xyz")
    history_rot = traj_data.get("ego_history_rot")
    if any(value is None for value in (future_xyz, future_rot, history_xyz, history_rot)):
        raise ValueError("future trajectory encoding requires history and future xyz/rot")
    if future_xyz.ndim != 4:
        raise ValueError("ego_future_xyz must have shape [B, n_traj, T, 3]")
    if future_rot.ndim != 5:
        raise ValueError("ego_future_rot must have shape [B, n_traj, T, 3, 3]")

    batch_size = future_xyz.shape[0]
    future_ids = tokenizer.encode(
        hist_xyz=history_xyz.flatten(start_dim=0, end_dim=1),
        hist_rot=history_rot.flatten(start_dim=0, end_dim=1),
        fut_xyz=future_xyz.flatten(start_dim=0, end_dim=1),
        fut_rot=future_rot.flatten(start_dim=0, end_dim=1),
    )
    future_ids = future_ids + start_idx
    return einops.rearrange(future_ids, "(b n_traj) n -> b (n_traj n)", b=batch_size)


class TrajectoryTokenFuser:
    """Parameter-free trajectory codec shared by ReasoningVLA and collators."""

    def __init__(
        self,
        *,
        traj_tokenizer: Any,
        hist_traj_tokenizer: Any,
        future_token_start_idx: int,
        history_token_start_idx: int,
        history_placeholder_id: int,
        future_placeholder_id: int,
    ) -> None:
        self.traj_tokenizer = traj_tokenizer
        self.hist_traj_tokenizer = hist_traj_tokenizer
        self.future_token_start_idx = future_token_start_idx
        self.history_token_start_idx = history_token_start_idx
        self.history_placeholder_id = history_placeholder_id
        self.future_placeholder_id = future_placeholder_id

    @classmethod
    def from_config(cls, config: Any) -> TrajectoryTokenFuser:
        if config.traj_tokenizer_cfg is None or config.hist_traj_tokenizer_cfg is None:
            raise ValueError("both future and history trajectory tokenizer configs are required")
        traj_tokenizer = instantiate(config.traj_tokenizer_cfg)
        hist_traj_tokenizer = instantiate(config.hist_traj_tokenizer_cfg)
        future_start = config.future_token_start_idx or config.traj_token_start_idx
        history_start = config.history_token_start_idx
        if history_start is None:
            history_start = future_start + traj_tokenizer.vocab_size
        try:
            history_placeholder = config.traj_token_ids["history"]
            future_placeholder = config.traj_token_ids["future"]
        except KeyError as exc:
            raise ValueError(f"missing trajectory placeholder ID: {exc.args[0]}") from exc
        return cls(
            traj_tokenizer=traj_tokenizer,
            hist_traj_tokenizer=hist_traj_tokenizer,
            future_token_start_idx=future_start,
            history_token_start_idx=history_start,
            history_placeholder_id=history_placeholder,
            future_placeholder_id=future_placeholder,
        )

    def fuse(
        self,
        input_ids: torch.Tensor,
        traj_data: Mapping[str, Any] | None,
    ) -> torch.Tensor:
        """Replace history and optional future placeholders without changing length."""

        if traj_data is None or traj_data.get("ego_history_xyz") is None or traj_data.get("ego_history_rot") is None:
            return input_ids
        history_ids = tokenize_history_trajectory(
            self.hist_traj_tokenizer,
            traj_data,
            self.history_token_start_idx,
        )
        fused = replace_placeholder_tokens(
            input_ids,
            history_ids,
            self.history_placeholder_id,
            component="history trajectory",
        )
        has_future = traj_data.get("ego_future_xyz") is not None and traj_data.get("ego_future_rot") is not None
        if has_future:
            future_ids = tokenize_future_trajectory(
                self.traj_tokenizer,
                traj_data,
                self.future_token_start_idx,
            )
            fused = replace_placeholder_tokens(
                fused,
                future_ids,
                self.future_placeholder_id,
                component="future trajectory",
            )
        return fused
