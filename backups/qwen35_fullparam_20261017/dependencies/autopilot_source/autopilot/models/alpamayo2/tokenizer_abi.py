# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared Alpamayo 1.5 trajectory-token ABI for the Qwen3.5 adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

SPECIAL_TOKEN_KEYS = (
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
)
SPECIAL_TOKENS = {key: f"<|{key}|>" for key in SPECIAL_TOKEN_KEYS}
TRAJ_TOKEN = {
    "history": SPECIAL_TOKENS["traj_history"],
    "future": SPECIAL_TOKENS["traj_future"],
    "history_start": SPECIAL_TOKENS["traj_history_start"],
    "future_start": SPECIAL_TOKENS["traj_future_start"],
    "history_end": SPECIAL_TOKENS["traj_history_end"],
    "future_end": SPECIAL_TOKENS["traj_future_end"],
}


def _validate_atomic_tokens(
    tokenizer: Any,
    tokens: Sequence[str],
    token_ids: Sequence[int],
    description: str,
) -> None:
    encoded = tokenizer(list(tokens), add_special_tokens=False)["input_ids"]
    for token, token_id, encoded_ids in zip(tokens, token_ids, encoded, strict=True):
        if encoded_ids != [token_id]:
            raise ValueError(f"{description} token {token!r} must encode to exactly [{token_id}], got {encoded_ids}")


def validate_tokenizer_abi(tokenizer: Any, traj_vocab_size: int | None) -> None:
    """Validate a prebuilt Alpamayo tokenizer without changing its vocabulary."""

    vocab = tokenizer.get_vocab()
    structured_tokens = list(SPECIAL_TOKENS.values())
    missing_structured = [token for token in structured_tokens if token not in vocab]
    if missing_structured:
        raise ValueError(
            "prebuilt tokenizer is missing Alpamayo 1.5 structured tokens: " + ", ".join(missing_structured)
        )
    structured_ids = [vocab[token] for token in structured_tokens]
    if len(set(structured_ids)) != len(structured_ids):
        raise ValueError("Alpamayo 1.5 structured tokens must have unique token IDs")
    _validate_atomic_tokens(tokenizer, structured_tokens, structured_ids, "structured")
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", {})
    for token, token_id in zip(structured_tokens, structured_ids, strict=True):
        metadata = added_tokens_decoder.get(token_id)
        if token_id not in special_ids and not getattr(metadata, "special", False):
            raise ValueError(f"structured token {token!r} must be registered as special")

    if traj_vocab_size is None:
        return
    if not isinstance(traj_vocab_size, int) or isinstance(traj_vocab_size, bool) or traj_vocab_size <= 0:
        raise ValueError("traj_vocab_size must be a positive integer or None")
    trajectory_tokens = [f"<i{index}>" for index in range(traj_vocab_size)]
    missing_trajectory = [token for token in trajectory_tokens if token not in vocab]
    if missing_trajectory:
        preview = ", ".join(missing_trajectory[:5])
        suffix = "..." if len(missing_trajectory) > 5 else ""
        raise ValueError(f"prebuilt tokenizer is missing trajectory tokens: {preview}{suffix}")
    trajectory_ids = [vocab[token] for token in trajectory_tokens]
    expected_ids = list(range(trajectory_ids[0], trajectory_ids[0] + traj_vocab_size))
    if trajectory_ids != expected_ids:
        raise ValueError("trajectory vocabulary IDs must be contiguous")
    _validate_atomic_tokens(tokenizer, trajectory_tokens, trajectory_ids, "trajectory")
    for token, token_id in zip(trajectory_tokens, trajectory_ids, strict=True):
        metadata = added_tokens_decoder.get(token_id)
        if token_id in special_ids or getattr(metadata, "special", False):
            raise ValueError(f"trajectory token {token!r} must not be registered as special")

    tokenizer.traj_token_start_idx = trajectory_ids[0]
    tokenizer.traj_token_end_idx = trajectory_ids[-1]
    tokenizer.traj_token_ids = {key: tokenizer.convert_tokens_to_ids(value) for key, value in TRAJ_TOKEN.items()}


def add_alpamayo_tokens(tokenizer: Any, traj_vocab_size: int) -> tuple[int, int]:
    """Add the Alpamayo vocabulary once and return trajectory/structured counts."""

    existing_vocab = tokenizer.get_vocab()
    if any(f"<i{index}>" in existing_vocab for index in range(traj_vocab_size)):
        raise ValueError("base tokenizer already contains Alpamayo trajectory tokens")
    trajectory_tokens = [f"<i{index}>" for index in range(traj_vocab_size)]
    trajectory_added = tokenizer.add_tokens(trajectory_tokens, special_tokens=False)
    if trajectory_added != traj_vocab_size:
        raise ValueError(f"expected to add {traj_vocab_size} trajectory tokens, added {trajectory_added}")
    structured_added = tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    validate_tokenizer_abi(tokenizer, traj_vocab_size)
    return trajectory_added, structured_added
