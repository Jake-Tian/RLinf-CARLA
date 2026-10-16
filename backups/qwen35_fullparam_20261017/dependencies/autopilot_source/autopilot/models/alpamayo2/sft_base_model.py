# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Trainable Stage-1 ReasoningVLA adapted to native Qwen3.5 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import einops
import numpy as np
import torch
from transformers import GenerationConfig
from transformers.utils import ModelOutput

from .base_model import (
    IGNORE_INDEX,
    ReasoningVLA,
    ReasoningVLAConfig,
    TrajectoryFusionMixin,
    load_alpamayo2_vlm,
)
from .token_utils import extract_text_tokens, extract_traj_tokens
from .trajectory_fuser import tokenize_future_trajectory as tokenize_future_trajectory


class TrajectoryFusionWithFutureMixin(TrajectoryFusionMixin):
    """Fuse history and, when supplied, teacher-forced future trajectories."""

    def fuse_traj_tokens(
        self,
        input_ids: torch.Tensor,
        traj_data: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        return TrajectoryFusionMixin.fuse_traj_tokens(self, input_ids, traj_data)


@dataclass
class ReasoningVLAOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    future_traj_loss: torch.FloatTensor | None = None
    other_loss: torch.FloatTensor | None = None


class TrainableReasoningVLA(ReasoningVLA, TrajectoryFusionWithFutureMixin):
    """Stage-1 SFT wrapper that loads a stitched native Qwen3.5 checkpoint."""

    fuse_traj_tokens = TrajectoryFusionWithFutureMixin.fuse_traj_tokens

    @classmethod
    def from_alpamayo_checkpoint(
        cls,
        checkpoint_path: str,
        vlm_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> TrainableReasoningVLA:
        """Load native Qwen3.5 weights and Alpamayo trajectory metadata.

        ``vlm_name_or_path`` is kept for recipe compatibility. The stitched
        checkpoint itself is always the processor and VLM source, so tokenizer
        IDs cannot drift away from the resized embedding rows.
        """

        checkpoint_dir = Path(checkpoint_path)
        if vlm_name_or_path is not None:
            supplied = Path(vlm_name_or_path)
            if supplied.exists() and supplied.resolve() != checkpoint_dir.resolve():
                raise ValueError(
                    "vlm_name_or_path must point to the stitched checkpoint; using a bare "
                    "Qwen checkpoint would break the trajectory-token ABI"
                )
        config = ReasoningVLAConfig.from_checkpoint(checkpoint_dir)
        print_param_count = kwargs.pop("print_param_count", True)
        loader_keys = {
            "attn_implementation",
            "device_map",
            "dtype",
            "low_cpu_mem_usage",
            "torch_dtype",
        }
        loader_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in loader_keys}
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported from_alpamayo_checkpoint arguments: {unknown}")
        vlm = load_alpamayo2_vlm(checkpoint_dir, **loader_kwargs)
        return cls(
            config,
            pretrained_modules={"vlm": vlm},
            original_vocab_size=config.original_vocab_size,
            print_param_count=print_param_count,
        )

    @torch._dynamo.disable
    def _compute_next_token_loss(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels_mask is None:
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
        if labels_mask[:, 1:].sum() == 0:
            return outputs.logits.new_zeros(())
        shift_labels = labels[..., 1:]
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
        shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
        shift_labels = shift_labels.to(shift_logits.device)
        if token_mask is not None:
            shift_logits = shift_logits.clone()
            shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
        return torch.nan_to_num(
            torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            ),
            nan=0.0,
        )

    def forward(
        self,
        tokenized_data: Mapping[str, Any] | None = None,
        input_ids: torch.Tensor | None = None,
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """Run Qwen3.5 without mutating the caller's tokenized-data mapping."""

        model_inputs = dict(tokenized_data or {})
        model_inputs.update(kwargs)
        if input_ids is None:
            input_ids = model_inputs.pop("input_ids", None)
        else:
            model_inputs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("input_ids must be passed directly or in tokenized_data")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        fused_input_ids = self.fuse_traj_tokens(input_ids, traj_data)
        model_inputs = self.prepare_multimodal_model_inputs(fused_input_ids, model_inputs)
        labels = fused_input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        outputs = self.vlm(input_ids=fused_input_ids, **model_inputs)
        supervised = labels != IGNORE_INDEX
        trajectory = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.traj_tokenizer.vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        ) & supervised
        future_loss = self._compute_next_token_loss(outputs, labels, trajectory)
        other_loss = self._compute_next_token_loss(outputs, labels, supervised & ~trajectory)
        loss = future_loss + other_loss
        return ReasoningVLAOutput(
            loss=loss,
            logits=outputs.logits,
            future_traj_loss=future_loss,
            other_loss=other_loss,
        )

    @torch.no_grad()
    def sample_trajectories_from_data(
        self,
        data: Mapping[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, list[str]]]:
        history_xyz = data["ego_history_xyz"]
        history_rot = data["ego_history_rot"]
        batch_size, trajectory_groups = history_xyz.shape[:2]
        if trajectory_groups != 1:
            raise ValueError("Only one trajectory group is supported for inference")
        if "tokenized_data" not in data:
            raise ValueError("sample_trajectories_from_data requires tokenized_data")
        tokenized_data = dict(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        input_ids = self.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": history_xyz,
                "ego_history_rot": history_rot,
            },
        )

        sample_count = num_traj_samples * num_traj_sets
        return_extra = kwargs.pop("return_extra", False)
        max_new_tokens = kwargs.pop("max_generation_length", self.config.tokens_per_future_traj + 1)
        generation_config = GenerationConfig.from_dict(self.vlm.generation_config.to_dict())
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = sample_count
        generation_config.max_new_tokens = max_new_tokens
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        if top_k is not None:
            generation_config.top_k = top_k
        tokenized_data = self.prepare_multimodal_model_inputs(input_ids, tokenized_data)
        generated = self.vlm.generate(
            input_ids=input_ids,
            **tokenized_data,
            **kwargs,
            generation_config=generation_config,
        )
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]
        trajectory_ids = extract_traj_tokens(
            generated_tokens,
            self.special_token_ids,
            self.config.tokens_per_future_traj,
            self.future_token_start_idx,
            self.traj_tokenizer.vocab_size,
        )
        repeated_history_xyz = einops.repeat(history_xyz[:, 0], "b ... -> (b n) ...", n=sample_count)
        repeated_history_rot = einops.repeat(history_rot[:, 0], "b ... -> (b n) ...", n=sample_count)
        prediction_xyz, prediction_rot, _ = self.traj_tokenizer.decode(
            hist_xyz=repeated_history_xyz,
            hist_rot=repeated_history_rot,
            tokens=trajectory_ids,
        )
        prediction_xyz = einops.rearrange(
            prediction_xyz,
            "(b ns nj) ... -> b ns nj ...",
            b=batch_size,
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        prediction_rot = einops.rearrange(
            prediction_rot,
            "(b ns nj) ... -> b ns nj ...",
            b=batch_size,
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        if return_extra:
            extra = extract_text_tokens(self.tokenizer, generated_tokens)
            for key, values in extra.items():
                extra[key] = np.asarray(values).reshape(batch_size, num_traj_sets, num_traj_samples)
            return prediction_xyz, prediction_rot, extra
        return prediction_xyz, prediction_rot
