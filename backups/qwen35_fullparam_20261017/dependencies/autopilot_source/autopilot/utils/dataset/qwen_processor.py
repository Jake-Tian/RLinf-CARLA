# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alpamayo Qwen preprocessing plus an adapter for VeRL's SFT batch schema."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch

from autopilot.models.alpamayo2.tokenizer_abi import (
    SPECIAL_TOKENS,
    validate_tokenizer_abi,
)
from autopilot.models.alpamayo2.trajectory_fuser import TrajectoryTokenFuser

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


def _model_config_value(model_config: Any, key: str, default: Any = None) -> Any:
    """Read Hydra DictConfig/dict/dataclass model settings uniformly."""

    if isinstance(model_config, dict):
        return model_config.get(key, default)
    return getattr(model_config, key, default)


def sort_images_by_camera_ids(
    image_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    relative_timestamps: torch.Tensor | None = None,
    return_camera_ids: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Stable-sort images and aligned metadata by camera ID."""

    order = torch.argsort(camera_indices, stable=True)
    result = [image_frames[order]]
    if return_camera_ids:
        result.append(camera_indices[order])
    if relative_timestamps is not None:
        result.append(relative_timestamps[order])
    return result[0] if len(result) == 1 else tuple(result)


def basic_collation_fn(
    batch: list[dict[str, Any]], unstackable_keys: Sequence[str] = ()
) -> dict[str, Any]:
    """Stack tensor fields and preserve other fields as per-sample lists."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    keys = set(batch[0])
    if any(set(row) != keys for row in batch):
        raise ValueError("all samples in a batch must contain the same keys")
    unstackable = set(unstackable_keys)
    return {
        key: torch.stack([row[key] for row in batch])
        if isinstance(batch[0][key], torch.Tensor) and key not in unstackable
        else [row[key] for row in batch]
        for key in keys
    }


def _component_text(name: str, content: str | None = None, count: int = 0, ask: bool = False) -> str:
    start = SPECIAL_TOKENS[f"{name}_start"]
    if ask:
        return start
    end = SPECIAL_TOKENS[f"{name}_end"]
    if content is not None:
        return f"{start}{content}{end}"
    return f"{start}{SPECIAL_TOKENS[name] * count}{end}"


def build_conversation(
    data: dict[str, Any],
    num_tokens_per_history_traj: int,
    num_tokens_per_future_traj: int,
    components_order: list[str],
    components_prompt: list[str],
    generation_mode: bool,
    include_camera_ids: bool,
    camera_ids: torch.Tensor,
    include_frame_nums: bool,
) -> list[dict[str, Any]]:
    """Build the R1.5 conversation used by Alpamayo trajectory and nav SFT."""

    if not components_order:
        raise ValueError("components_order must not be empty")
    user_content: list[dict[str, Any]] = []
    assistant_content: list[dict[str, Any]] = []
    last_component = components_order[-1]
    prompt_names = {
        "cot": "output the chain-of-thought reasoning of the driving process",
        "meta_action": "output meta actions",
        "traj_future": "output the future trajectory",
    }

    for component in components_order:
        ask = generation_mode and component == last_component
        if component == "prompt":
            selected = components_prompt if generation_mode else components_order
            prompt = ", then ".join(prompt_names[name] for name in selected if name in prompt_names) + "."
            user_content.append({"type": "text", "text": prompt})
        elif component == "image":
            if not torch.equal(camera_ids, torch.sort(camera_ids).values):
                raise ValueError("camera IDs must be sorted before building the conversation")
            for camera_id, view_images in zip(camera_ids.tolist(), data["image_frames"], strict=True):
                if include_camera_ids:
                    user_content.append({"type": "text", "text": f"{CAMERA_DISPLAY_NAMES[camera_id]}: "})
                for frame_number, frame in enumerate(view_images):
                    if include_frame_nums:
                        user_content.append({"type": "text", "text": f"frame {frame_number} "})
                    user_content.append({"type": "image", "image": frame})
        elif component == "traj_history":
            user_content.append(
                {
                    "type": "text",
                    "text": _component_text("traj_history", count=num_tokens_per_history_traj),
                }
            )
        elif component == "route":
            if data.get("nav_text") is not None:
                user_content.append(
                    {"type": "text", "text": _component_text("route", content=str(data["nav_text"]))}
                )
        elif component in {"cot", "meta_action", "traj_future", "answer"}:
            if component == "traj_future":
                text = _component_text(component, count=num_tokens_per_future_traj, ask=ask)
            else:
                field = {"cot": "cot", "meta_action": "meta_action_strings", "answer": "answer"}[component]
                if not ask and field not in data:
                    raise ValueError(f"{field!r} is required by components_order")
                text = _component_text(component, content=None if ask else str(data[field]), ask=ask)
            assistant_content.append({"type": "text", "text": text})
        elif component == "question":
            if not ask and "question" not in data:
                raise ValueError("'question' is required by components_order")
            user_content.append(
                {
                    "type": "text",
                    "text": _component_text("question", content=None if ask else str(data["question"]), ask=ask),
                }
            )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _fill_between_tokens(
    input_ids: torch.Tensor, tokenizer: Any, start_token: str, end_token: str, mask: torch.Tensor
) -> None:
    start_id = tokenizer.convert_tokens_to_ids(start_token)
    end_id = tokenizer.convert_tokens_to_ids(end_token)
    for batch_index in range(input_ids.shape[0]):
        starts = torch.where(input_ids[batch_index] == start_id)[0]
        ends = torch.where(input_ids[batch_index] == end_id)[0]
        if len(starts) != len(ends):
            raise ValueError(f"Unpaired component tokens {start_token!r}/{end_token!r}")
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            if end < start:
                raise ValueError(f"Component end precedes start for {start_token!r}")
            mask[batch_index, start : end + 1] = True


def get_label_mask(input_ids: torch.Tensor, tokenizer: Any, label_components: list[str]) -> torch.Tensor:
    """Select complete structured components as SFT targets."""

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for component in label_components:
        start_key, end_key = f"{component}_start", f"{component}_end"
        if start_key not in SPECIAL_TOKENS or end_key not in SPECIAL_TOKENS:
            raise ValueError(f"Unknown label component {component!r}")
        _fill_between_tokens(input_ids, tokenizer, SPECIAL_TOKENS[start_key], SPECIAL_TOKENS[end_key], mask)
    return mask


def get_role_eos_mask(
    input_ids: torch.Tensor,
    tokenizer: Any,
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>",
    role: str = "assistant",
) -> torch.Tensor:
    """Select only EOS markers which terminate an assistant block."""

    bos_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    role_id = tokenizer.convert_tokens_to_ids(role)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for batch_index in range(input_ids.shape[0]):
        starts = torch.where(input_ids[batch_index] == bos_id)[0].tolist()
        ends = torch.where(input_ids[batch_index] == eos_id)[0].tolist()
        for start, end in zip(starts, ends, strict=False):
            if start + 1 < input_ids.shape[1] and input_ids[batch_index, start + 1] == role_id:
                mask[batch_index, end] = True
    return mask


class QwenProcessor:
    """Lazy Hugging Face processor with Alpamayo vocabulary and prompt semantics."""

    def __init__(
        self,
        vlm_name_or_path: str,
        traj_vocab_size: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
        chat_template_version: str = "r1_5",
        trust_remote_code: bool = False,
    ) -> None:
        if chat_template_version not in {"r1_5", "r1.5"}:
            raise ValueError(
                f"Only the Alpamayo 1.5 chat template is supported, got {chat_template_version!r}"
            )
        self.vlm_name_or_path = vlm_name_or_path
        self.traj_vocab_size = traj_vocab_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums
        self.trust_remote_code = trust_remote_code
        self._processor: Any | None = None

    def build_processor(self) -> Any:
        from transformers import AutoProcessor

        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.min_pixels is not None:
            kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            kwargs["max_pixels"] = self.max_pixels
        processor = AutoProcessor.from_pretrained(self.vlm_name_or_path, **kwargs)
        if (
            not hasattr(processor, "image_processor")
            or not hasattr(processor, "tokenizer")
            or not isinstance(getattr(processor, "image_token", None), str)
        ):
            raise ValueError(f"{self.vlm_name_or_path!r} does not provide a multimodal processor")
        tokenizer = processor.tokenizer
        validate_tokenizer_abi(tokenizer, self.traj_vocab_size)
        return processor

    @property
    def processor(self) -> Any:
        if self._processor is None:
            self._processor = self.build_processor()
        return self._processor

    def get_preprocess_data_fn(
        self,
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
        **_: Any,
    ) -> Callable[..., Any]:
        return partial(
            self._preprocess_data,
            processor=self.processor,
            num_tokens_per_history_traj=num_tokens_per_history_traj,
            num_tokens_per_future_traj=num_tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            label_components=label_components,
            generation_mode=generation_mode,
        )

    def _preprocess_data(
        self,
        data: dict[str, Any],
        processor: Any,
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
    ) -> dict[str, Any]:
        images, camera_ids, timestamps = sort_images_by_camera_ids(
            data["image_frames"],
            data["camera_indices"],
            relative_timestamps=data["relative_timestamps"],
            return_camera_ids=True,
        )
        data["image_frames"] = images
        data["camera_indices"] = camera_ids
        data["relative_timestamps"] = timestamps
        data["label_components"] = label_components
        data["generation_mode"] = generation_mode
        messages = build_conversation(
            data=data,
            num_tokens_per_history_traj=num_tokens_per_history_traj,
            num_tokens_per_future_traj=num_tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            generation_mode=generation_mode,
            include_camera_ids=self.include_camera_ids,
            camera_ids=camera_ids,
            include_frame_nums=self.include_frame_nums,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=generation_mode,
        )
        images_for_vlm = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
        image_inputs = processor.image_processor(images=images_for_vlm.flatten(0, 1), do_rescale=False)
        merge_length = processor.image_processor.merge_size**2
        image_index = 0
        while processor.image_token in text:
            token_count = image_inputs["image_grid_thw"][image_index].prod() // merge_length
            text = text.replace(processor.image_token, "<|placeholder|>" * int(token_count), 1)
            image_index += 1
        if image_index != len(image_inputs["image_grid_thw"]):
            raise ValueError("chat template image placeholders do not match processed images")
        return {"text": text.replace("<|placeholder|>", processor.image_token), **image_inputs}

    def collate_fn(self, data: list[dict[str, Any]], padding_side: str = "left") -> dict[str, Any]:
        """Return the original Alpamayo/Hugging Face ReasoningVLA batch."""

        batched = basic_collation_fn(data, unstackable_keys=("image_frames",))
        tokenized_rows = batched["tokenized_data"]
        tokenized = {
            key: torch.cat([row[key] for row in tokenized_rows])
            for key in tokenized_rows[0]
            if key != "text"
        }
        tokenized.update(
            self.processor.tokenizer(
                [row["text"] for row in tokenized_rows],
                return_tensors="pt",
                padding_side=padding_side,
                padding=True,
            )
        )
        batched["tokenized_data"] = tokenized
        label_components = batched["label_components"][0]
        generation_mode = batched["generation_mode"][0]
        if any(value != label_components for value in batched["label_components"]):
            raise ValueError("label_components must be identical within a batch")
        if any(value != generation_mode for value in batched["generation_mode"]):
            raise ValueError("generation_mode must be identical within a batch")
        if generation_mode:
            batched["labels_mask"] = torch.zeros_like(tokenized["input_ids"], dtype=torch.bool)
        else:
            batched["labels_mask"] = get_label_mask(
                tokenized["input_ids"], self.processor.tokenizer, label_components
            )
            batched["labels_mask"] |= get_role_eos_mask(tokenized["input_ids"], self.processor.tokenizer)
        return batched


def build_qwen_vl_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    """Match VeRL's four-row text + Qwen vision position-ID convention."""

    from verl.models.transformers.qwen2_vl import get_rope_index

    vision_positions = get_rope_index(
        processor,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    text_positions = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)
    return torch.cat((text_positions, vision_positions), dim=0)


class AlpamayoSFTCollator:
    """Fuse Alpamayo preprocessing with VeRL's no-padding SFT collation contract."""

    def __init__(
        self,
        processor: QwenProcessor,
        trajectory_fuser: TrajectoryTokenFuser,
        padding_side: str = "left",
        position_ids_builder: Callable[..., torch.Tensor] = build_qwen_vl_position_ids,
    ) -> None:
        self.qwen_processor = processor
        self.trajectory_fuser = trajectory_fuser
        self.padding_side = padding_side
        self.position_ids_builder = position_ids_builder

    def __call__(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator

        alpamayo_batch = self.qwen_processor.collate_fn(data, padding_side=self.padding_side)
        tokenized = alpamayo_batch["tokenized_data"]
        traj_data: dict[str, torch.Tensor] = {}
        for key in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot"):
            if key not in alpamayo_batch:
                raise ValueError(f"Alpamayo SFT batch is missing required trajectory field {key!r}")
            value = alpamayo_batch[key]
            if key.endswith("xyz") and value.ndim == 3:
                value = value.unsqueeze(1)
            elif key.endswith("rot") and value.ndim == 4:
                value = value.unsqueeze(1)
            traj_data[key] = value
        tokenized["input_ids"] = self.trajectory_fuser.fuse(tokenized["input_ids"], traj_data)
        per_sample: list[dict[str, Any]] = []
        for index, row in enumerate(data):
            valid = tokenized["attention_mask"][index].bool()
            input_ids = tokenized["input_ids"][index][valid]
            attention_mask = torch.ones_like(input_ids)
            image_grid_thw = row["tokenized_data"].get("image_grid_thw")
            position_ids = self.position_ids_builder(
                self.qwen_processor.processor,
                input_ids,
                attention_mask,
                image_grid_thw,
            )
            multi_modal_inputs = {
                key: value for key, value in row["tokenized_data"].items() if key != "text"
            }
            per_sample.append(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "loss_mask": alpamayo_batch["labels_mask"][index][valid],
                    "multi_modal_inputs": multi_modal_inputs,
                }
            )
        batch = SFTTensorCollator(DatasetPadMode.NO_PADDING)(per_sample)
        return batch


def get_preprocess_data_fn_from_model_config(
    components_order: list[str] | None = None,
    components_prompt: list[str] | None = None,
    label_components: list[str] | None = None,
    generation_mode: bool = False,
    include_camera_ids: bool = False,
    include_frame_nums: bool = False,
    model_config: Any = None,
    chat_template_version: str = "r1_5",
    **kwargs: Any,
) -> Callable[..., Any]:
    """Hydra factory for per-sample preprocessing."""

    qwen_processor = QwenProcessor(
        vlm_name_or_path=_model_config_value(model_config, "vlm_name_or_path"),
        traj_vocab_size=_model_config_value(model_config, "traj_vocab_size"),
        min_pixels=_model_config_value(model_config, "min_pixels"),
        max_pixels=_model_config_value(model_config, "max_pixels"),
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
        chat_template_version=chat_template_version,
        trust_remote_code=_model_config_value(model_config, "trust_remote_code", False),
    )
    return qwen_processor.get_preprocess_data_fn(
        num_tokens_per_history_traj=_model_config_value(model_config, "tokens_per_history_traj"),
        num_tokens_per_future_traj=_model_config_value(model_config, "tokens_per_future_traj"),
        components_order=components_order or [],
        components_prompt=components_prompt or [],
        label_components=label_components or [],
        generation_mode=generation_mode,
        **kwargs,
    )


def collate_fn_from_model_config(
    model_config: Any,
    padding_side: str = "left",
    include_camera_ids: bool = False,
    include_frame_nums: bool = False,
    chat_template_version: str = "r1_5",
) -> AlpamayoSFTCollator:
    """Hydra factory returning the Autopilot/VeRL batch adapter."""

    processor = QwenProcessor(
        vlm_name_or_path=_model_config_value(model_config, "vlm_name_or_path"),
        traj_vocab_size=_model_config_value(model_config, "traj_vocab_size"),
        min_pixels=_model_config_value(model_config, "min_pixels"),
        max_pixels=_model_config_value(model_config, "max_pixels"),
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
        chat_template_version=chat_template_version,
        trust_remote_code=_model_config_value(model_config, "trust_remote_code", False),
    )
    from autopilot.models.alpamayo2.base_model import ReasoningVLAConfig

    checkpoint_path = _model_config_value(model_config, "vlm_name_or_path")
    trajectory_fuser = TrajectoryTokenFuser.from_config(
        ReasoningVLAConfig.from_checkpoint(checkpoint_path)
    )
    return AlpamayoSFTCollator(
        processor=processor,
        trajectory_fuser=trajectory_fuser,
        padding_side=padding_side,
    )
