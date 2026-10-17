# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-backed Stage-1 ReasoningVLA domain wrapper."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra.utils import instantiate
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
)

from .tokenizer_abi import SPECIAL_TOKENS, TRAJ_TOKEN, validate_tokenizer_abi
from .trajectory_fuser import (
    TrajectoryTokenFuser,
    replace_placeholder_tokens,
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
REASONING_VLA_CONFIG_NAME = "reasoning_vla_config.json"
REASONING_VLA_NAMESPACE = "autopilot_reasoning_vla"


# Backward-compatible name used by the upstream recipe.
def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return replace_placeholder_tokens(input_ids, new_ids, pad_idx, component="trajectory")


class TrajectoryFusionMixin:
    """Fuse encoded history placeholders into a tokenized prompt."""

    def _validate_mixin_requirements(self, require_future: bool = False) -> dict[str, Any]:
        required = {
            "hist_traj_tokenizer": getattr(self, "hist_traj_tokenizer", None),
            "hist_token_start_idx": getattr(self, "hist_token_start_idx", None),
            "config": getattr(self, "config", None),
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise AttributeError("TrajectoryFusionMixin is missing: " + ", ".join(missing))
        if not hasattr(required["config"], "traj_token_ids"):
            raise AttributeError("ReasoningVLA config must define traj_token_ids")
        if require_future:
            for key in ("traj_tokenizer", "future_token_start_idx"):
                value = getattr(self, key, None)
                if value is None:
                    raise AttributeError(f"TrajectoryFusionMixin is missing: {key}")
                required[key] = value
        return required

    def fuse_traj_tokens(
        self,
        input_ids: torch.Tensor,
        traj_data: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if traj_data is None or traj_data.get("ego_history_xyz") is None or traj_data.get("ego_history_rot") is None:
            return input_ids
        attrs = self._validate_mixin_requirements(require_future=False)
        fuser = getattr(self, "trajectory_token_fuser", None)
        if fuser is None:
            fuser = TrajectoryTokenFuser(
                traj_tokenizer=getattr(self, "traj_tokenizer", attrs["hist_traj_tokenizer"]),
                hist_traj_tokenizer=attrs["hist_traj_tokenizer"],
                future_token_start_idx=getattr(self, "future_token_start_idx", attrs["hist_token_start_idx"]),
                history_token_start_idx=attrs["hist_token_start_idx"],
                history_placeholder_id=attrs["config"].traj_token_ids["history"],
                future_placeholder_id=attrs["config"].traj_token_ids["future"],
            )
        return fuser.fuse(input_ids, traj_data)


class ReasoningVLAConfig(PretrainedConfig):
    """Wrapper config loaded from native Qwen3.5 checkpoint metadata."""

    model_type = "autopilot_alpamayo2"

    def __init__(
        self,
        vlm_name_or_path: str | None = None,
        vlm_backend: str = "qwen3_5",
        traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        traj_vocab_size: int = 4000,
        tokens_per_history_traj: int = 48,
        tokens_per_future_traj: int = 128,
        traj_token_start_idx: int | None = None,
        traj_token_ids: dict[str, int] | None = None,
        special_token_ids: dict[str, int] | None = None,
        future_token_start_idx: int | None = None,
        history_token_start_idx: int | None = None,
        original_vocab_size: int | None = None,
        expanded_vocab_size: int | None = None,
        model_dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        abi_version: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_backend = vlm_backend
        self.traj_tokenizer_cfg = traj_tokenizer_cfg
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg
        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.traj_token_start_idx = traj_token_start_idx
        self.traj_token_ids = traj_token_ids or {}
        self.special_token_ids = special_token_ids or {}
        self.future_token_start_idx = future_token_start_idx
        self.history_token_start_idx = history_token_start_idx
        self.original_vocab_size = original_vocab_size
        self.expanded_vocab_size = expanded_vocab_size
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.abi_version = abi_version

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> ReasoningVLAConfig:
        checkpoint_dir = Path(checkpoint_path)
        sidecar_path = checkpoint_dir / REASONING_VLA_CONFIG_NAME
        native_config_path = checkpoint_dir / "config.json"
        if sidecar_path.exists():
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        elif native_config_path.exists():
            native = json.loads(native_config_path.read_text(encoding="utf-8"))
            metadata = native.get(REASONING_VLA_NAMESPACE)
            if metadata is None:
                raise ValueError(f"{native_config_path} has no {REASONING_VLA_NAMESPACE!r} metadata")
        else:
            raise FileNotFoundError(f"Missing config file under {checkpoint_dir}")
        metadata = dict(metadata)
        metadata["vlm_name_or_path"] = str(checkpoint_dir)
        return cls(**metadata)


def _native_checkpoint_weight_keys(checkpoint_dir: Path) -> list[str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty checkpoint index: {index_path}")
        missing_shards = sorted({shard for shard in weight_map.values() if not (checkpoint_dir / shard).is_file()})
        if missing_shards:
            raise FileNotFoundError("Checkpoint index refers to missing shards: " + ", ".join(missing_shards))
        return list(weight_map)
    single_file = checkpoint_dir / "model.safetensors"
    if single_file.exists():
        from safetensors import safe_open

        with safe_open(single_file, framework="pt", device="cpu") as handle:
            return list(handle.keys())
    raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json under {checkpoint_dir}")


def load_alpamayo2_vlm(
    checkpoint_path: str | Path,
    **model_kwargs: Any,
) -> PreTrainedModel:
    """Load the stitched checkpoint using its native Qwen3.5 key layout."""

    checkpoint_dir = Path(checkpoint_path)
    native_config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    if native_config.model_type != "qwen3_5":
        raise ValueError(f"Alpamayo2 expects a native qwen3_5 checkpoint, got {native_config.model_type!r}")
    weight_keys = _native_checkpoint_weight_keys(checkpoint_dir)
    prefixed = [key for key in weight_keys if key.startswith("vlm.")]
    if prefixed:
        raise ValueError("Alpamayo2 checkpoint must preserve native Qwen3.5 keys; found vlm.* keys")
    return AutoModelForImageTextToText.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        **model_kwargs,
    )


class ReasoningVLA(PreTrainedModel, TrajectoryFusionMixin):
    """Thin ReasoningVLA wrapper around a native Qwen3.5 VLM."""

    config_class = ReasoningVLAConfig
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, Any] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        super().__init__(config)
        pretrained_modules = pretrained_modules or {}
        self._initialize_vlm_backbone(config, pretrained_modules)
        self._initialize_trajectory_tokenizers(config, pretrained_modules)
        self.trajectory_token_fuser = TrajectoryTokenFuser(
            traj_tokenizer=self.traj_tokenizer,
            hist_traj_tokenizer=self.hist_traj_tokenizer,
            future_token_start_idx=self.future_token_start_idx,
            history_token_start_idx=self.hist_token_start_idx,
            history_placeholder_id=config.traj_token_ids["history"],
            future_placeholder_id=config.traj_token_ids["future"],
        )
        self.processor = AutoProcessor.from_pretrained(
            config.vlm_name_or_path,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        validate_tokenizer_abi(self.tokenizer, config.traj_vocab_size)
        self.special_token_ids = {
            key: self.tokenizer.convert_tokens_to_ids(value) for key, value in SPECIAL_TOKENS.items()
        }
        self.original_vocab_size = original_vocab_size or config.original_vocab_size
        self._validate_checkpoint_abi()
        if print_param_count:
            total_params = sum(parameter.numel() for parameter in self.parameters())
            trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            logger.info(
                "ReasoningVLA parameters: total=%s trainable=%s",
                f"{total_params:,}",
                f"{trainable_params:,}",
            )

    def _initialize_vlm_backbone(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, Any],
    ) -> None:
        if "vlm" in pretrained_modules:
            self.vlm = pretrained_modules["vlm"]
            return
        native_config = AutoConfig.from_pretrained(config.vlm_name_or_path, local_files_only=True)
        self.vlm = AutoModelForImageTextToText.from_config(native_config)

    def _initialize_trajectory_tokenizers(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, Any],
    ) -> None:
        if "traj_tokenizer" in pretrained_modules:
            self.traj_tokenizer = pretrained_modules["traj_tokenizer"]
        elif config.traj_tokenizer_cfg is not None:
            self.traj_tokenizer = instantiate(config.traj_tokenizer_cfg)
        else:
            self.traj_tokenizer = None

        if config.hist_traj_tokenizer_cfg is not None:
            self.hist_traj_tokenizer = instantiate(config.hist_traj_tokenizer_cfg)
        else:
            self.hist_traj_tokenizer = self.traj_tokenizer

        self.future_token_start_idx = config.future_token_start_idx or config.traj_token_start_idx
        if config.history_token_start_idx is not None:
            self.hist_token_start_idx = config.history_token_start_idx
        elif self.traj_tokenizer is not None:
            self.hist_token_start_idx = config.traj_token_start_idx + self.traj_tokenizer.vocab_size
        else:
            self.hist_token_start_idx = config.traj_token_start_idx

    def _validate_checkpoint_abi(self) -> None:
        tokenizer_size = len(self.tokenizer)
        embedding_size = self.vlm.get_input_embeddings().num_embeddings
        if tokenizer_size != embedding_size:
            raise ValueError(
                f"tokenizer/model vocabulary mismatch: tokenizer={tokenizer_size}, embedding={embedding_size}"
            )
        if self.config.expanded_vocab_size is not None and tokenizer_size != self.config.expanded_vocab_size:
            raise ValueError("reasoning VLA metadata expanded_vocab_size does not match tokenizer")
        actual_traj_ids = {key: self.tokenizer.convert_tokens_to_ids(value) for key, value in TRAJ_TOKEN.items()}
        if self.config.traj_token_ids and actual_traj_ids != self.config.traj_token_ids:
            raise ValueError("trajectory special-token IDs do not match checkpoint metadata")
        if self.traj_tokenizer is None or self.hist_traj_tokenizer is None:
            raise ValueError("both future and history trajectory tokenizers are required")
        future_end = self.future_token_start_idx + self.traj_tokenizer.vocab_size
        history_end = self.hist_token_start_idx + self.hist_traj_tokenizer.vocab_size
        traj_end = self.config.traj_token_start_idx + self.config.traj_vocab_size
        if self.future_token_start_idx != self.config.traj_token_start_idx:
            raise ValueError("future trajectory token block must start at <i0>")
        if self.hist_token_start_idx != future_end:
            raise ValueError("history trajectory token block must follow the future block")
        if history_end != traj_end:
            raise ValueError("future/history tokenizer vocab sizes do not cover traj_vocab_size")

    def prepare_multimodal_model_inputs(
        self,
        input_ids: torch.Tensor,
        model_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Add the Qwen3.5 M-RoPE token types for direct HF calls when needed."""

        prepared = dict(model_inputs)
        has_multimodal_grid = any(
            prepared.get(key) is not None for key in ("image_grid_thw", "video_grid_thw", "audio_grid_thw")
        )
        if (
            not has_multimodal_grid
            or prepared.get("position_ids") is not None
            or prepared.get("mm_token_type_ids") is not None
        ):
            return prepared
        create_token_types = getattr(self.processor, "create_mm_token_type_ids", None)
        if create_token_types is None:
            raise ValueError("multimodal Qwen3.5 inputs require mm_token_type_ids or position_ids")
        token_types = create_token_types(input_ids.detach().cpu().tolist())
        prepared["mm_token_type_ids"] = torch.tensor(
            token_types,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return prepared

    @classmethod
    def from_pretrained_submodules(
        cls,
        config: ReasoningVLAConfig,
        **model_kwargs: Any,
    ) -> ReasoningVLA:
        vlm = load_alpamayo2_vlm(config.vlm_name_or_path, **model_kwargs)
        return cls(
            config,
            pretrained_modules={"vlm": vlm},
            original_vocab_size=config.original_vocab_size,
        )

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.vlm.get_output_embeddings()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.vlm.get_input_embeddings()

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.vlm.gradient_checkpointing_disable()

    def generate(self, input_ids: torch.Tensor | None = None, **kwargs: Any) -> Any:
        if input_ids is None:
            input_ids = kwargs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("generate requires input_ids")
        model_inputs = self.prepare_multimodal_model_inputs(input_ids, kwargs)
        return self.vlm.generate(input_ids=input_ids, **model_inputs)
