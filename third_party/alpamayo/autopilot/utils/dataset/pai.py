# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch datasets built on the PAI local interface and geometry loader."""

from __future__ import annotations

import concurrent
import hydra.utils as hyu
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
import os
import torch
from torch.utils.data import Dataset, Subset, IterableDataset, get_worker_info

from autopilot.utils.dataset.pai_contract import PAIDataContract
from autopilot.utils.dataset.pai_loader import load_physical_aiavdataset
from autopilot.utils.dataset.pai_storage import PhysicalAIAVDatasetLocalInterface

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MANDATORY_PATTERNS = [
    "features.csv",
    "clip_index.parquet",
    "metadata/**",
]
OPTIONAL_COMPONENTS = ("camera", "calibration", "labels", "lidar", "radar")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def _instantiate_preprocessor(config: dict[str, Any], model_config: Any) -> Callable[..., Any]:
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError("hydra-core and omegaconf are required for vla_preprocess_args") from exc
    if isinstance(model_config, dict):
        model_config = OmegaConf.create(model_config)
    return instantiate(config, model_config=model_config)

def instantiate_dataset(dataset_config, model_config, max_samples: int = -1):
    """Instantiate a PAI dataset and apply an optional deterministic sample limit."""

    dataset = hyu.instantiate(dataset_config, _convert_="partial", model_config=model_config)
    if max_samples is not None and max_samples >= 0 and len(dataset) > max_samples:
        dataset = Subset(dataset, range(max_samples))
    return dataset


class PAIDataset(Dataset):
    """Lazy PAI dataset whose sample identity is ``(clip_id, t0_us)``."""

    DEFAULT_T0_US = 5_100_000

    def __init__(
        self,
        local_dir: str | Path | None = None,
        chunk_ids: str | int | list[int] | None = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        reasoning_metadata: str | None = None,
        split: str | None = None,
        default_t0_us: int | None = DEFAULT_T0_US,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        camera_features: list[str] | tuple[str, ...] | None = None,
        num_frames: int = 4,
        include_extr_intr: bool = False,
        reshape_tensors_for_rl: bool = False,
        validate_samples: bool = True,
        preprocessor: Callable[..., Any] | None = None,
        model_config: Any | None = None,
        vla_preprocess_args: dict[str, Any] | None = None,
        avdi: Any | None = None,
    ) -> None:
        """Create a dataset without reading video or egomotion payloads.

        ``avdi`` and ``preprocessor`` are injectable to keep unit tests and
        downstream model adapters independent from optional PAI dependencies.
        ``include_extr_intr`` retains the recipes' combined calibration/vehicle
        switch for configuration compatibility.
        """

        if avdi is None:
            if local_dir is None:
                raise ValueError("local_dir is required when avdi is not supplied")
            avdi = PhysicalAIAVDatasetLocalInterface(
                local_dir=local_dir,
                chunk_ids=chunk_ids,
                features_metadata=features_metadata,
                clip_index_metadata=clip_index_metadata,
                reasoning_metadata=reasoning_metadata,
                split=split,
            )
        elif chunk_ids is not None:
            raise ValueError("chunk_ids must be configured on an injected avdi, not PAIDataset")
        self.avdi = avdi

        contract_kwargs: dict[str, Any] = {
            "num_history_steps": num_history_steps,
            "num_future_steps": num_future_steps,
            "time_step_s": time_step,
            "num_frames": num_frames,
        }
        if camera_features is not None:
            contract_kwargs["camera_features"] = tuple(camera_features)
        self.contract = PAIDataContract(**contract_kwargs)
        self.anchors = self.avdi.get_sample_anchors(default_t0_us)
        if len(set(self.anchors)) != len(self.anchors):
            raise ValueError("PAI sample anchors must be unique (clip_id, t0_us) pairs")

        self.include_extr_intr = include_extr_intr
        self.reshape_tensors_for_rl = reshape_tensors_for_rl
        self.validate_samples = validate_samples
        if preprocessor is not None and vla_preprocess_args is not None:
            raise ValueError("Specify either preprocessor or vla_preprocess_args, not both")
        self.preprocessor = preprocessor
        if vla_preprocess_args is not None:
            self.preprocessor = _instantiate_preprocessor(vla_preprocess_args, model_config)

    @property
    def clip_ids(self) -> list[str]:
        """Compatibility view; may contain duplicates for multiple event anchors."""

        return [clip_id for clip_id, _ in self.anchors]

    def __len__(self) -> int:
        return len(self.anchors)

    def _load_anchor(self, clip_id: str, t0_us: int) -> dict[str, Any]:
        sample = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=t0_us,
            avdi=self.avdi,
            contract=self.contract,
            validate=self.validate_samples,
        )
        if self.include_extr_intr:
            sample["extr"] = self.avdi.get_clip_feature(clip_id, "sensor_extrinsics")
            sample["intr"] = self.avdi.get_clip_feature(clip_id, "camera_intrinsics")
            dimensions = self.avdi.get_clip_feature(clip_id, "vehicle_dimensions")
            length = float(dimensions.length)
            if length <= 0:
                raise ValueError(f"Clip {clip_id!r} has a non-positive vehicle length")
            sample["ego_lwh"] = torch.tensor(
                [dimensions.length, dimensions.width, dimensions.height], dtype=torch.float32
            )
            sample["ego_length_offset"] = torch.tensor(
                dimensions.rear_axle_to_bbox_center / length, dtype=torch.float32
            )
        if getattr(self.avdi, "reasoning_db", None) is not None:
            sample.update(self.avdi.get_reasoning_data(clip_id, t0_us))
        return sample

    @staticmethod
    def _reshape_for_rl(sample: dict[str, Any]) -> None:
        n_cam, n_frame = sample["image_frames"].shape[:2]
        sample["image_frames"] = sample["image_frames"].reshape(
            n_cam * n_frame, *sample["image_frames"].shape[2:]
        ).unsqueeze(1)
        sample["camera_indices"] = sample["camera_indices"].repeat_interleave(n_frame)
        sample["absolute_timestamps"] = sample["absolute_timestamps"].reshape(-1)
        sample["relative_timestamps"] = sample["relative_timestamps"].reshape(-1)

    def _finish_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.reshape_tensors_for_rl:
            self._reshape_for_rl(sample)
        if self.preprocessor is not None:
            sample["tokenized_data"] = self.preprocessor(data=sample)
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        clip_id, t0_us = self.anchors[index]
        return self._finish_sample(self._load_anchor(clip_id, t0_us))


class PAIDatasetWithNav(PAIDataset):
    """PAI dataset enumerated by JSON navigation annotations."""

    def __init__(self, annotations_path: str | Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        with Path(annotations_path).open(encoding="utf-8") as file:
            annotations = json.load(file)
        if not isinstance(annotations, list):
            raise ValueError("navigation annotations must be a JSON list")

        allowed = set(self.avdi.get_all_clip_ids())
        samples: list[dict[str, Any]] = []
        for index, entry in enumerate(annotations):
            if not isinstance(entry, dict) or "clip_id" not in entry:
                raise ValueError(f"annotation {index} must be an object with clip_id")
            timestamp_key = "t0_relative" if "t0_relative" in entry else "t0"
            if timestamp_key not in entry:
                raise ValueError(f"annotation {index} must contain t0_relative (or legacy t0)")
            if entry["clip_id"] in allowed:
                normalized = dict(entry)
                normalized["t0_us"] = int(entry[timestamp_key])
                samples.append(normalized)
        if not samples:
            raise ValueError("all navigation annotations were filtered out by the selected clip index/chunks")
        if len(samples) != len(annotations):
            logger.warning(
                "Filtered %d navigation annotations outside the selected clips", len(annotations) - len(samples)
            )
        anchors = [(str(entry["clip_id"]), entry["t0_us"]) for entry in samples]
        if len(set(anchors)) != len(anchors):
            raise ValueError("navigation annotations contain duplicate (clip_id, t0_us) anchors")
        self.annotations = samples
        self.anchors = anchors

    def __getitem__(self, index: int) -> dict[str, Any]:
        annotation = self.annotations[index]
        sample = self._load_anchor(str(annotation["clip_id"]), int(annotation["t0_us"]))
        reserved = {"clip_id", "t0", "t0_relative", "t0_us"}
        sample.update({key: value for key, value in annotation.items() if key not in reserved})
        return self._finish_sample(sample)


class IterablePAIDataset(IterableDataset):
    """Iterable PAI dataset whose sample identity is ``(clip_id, t0_us)``."""

    DEFAULT_T0_US = 5_100_000

    def __init__(
        self,
        local_dir: str | Path,
        chunk_ids: str | int | list[int] | None = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        reasoning_metadata: str | None = None,
        split: str | None = None,
        default_t0_us: int | None = DEFAULT_T0_US,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        camera_features: list[str] | tuple[str, ...] | None = None,
        num_frames: int = 4,
        include_extr_intr: bool = False,
        reshape_tensors_for_rl: bool = False,
        validate_samples: bool = True,
        preprocessor: Callable[..., Any] | None = None,
        model_config: Any | None = None,
        vla_preprocess_args: dict[str, Any] | None = None,
        cache_base_dir: str | Path = "/tmp",
        shuffle_seed: int = 42,
        avdi: Any | None = None,
    ) -> None:
        if avdi is not None:
            raise ValueError("avdi injection is not supported for IterablePAIDataset")
        self.local_dir = Path(local_dir).resolve()
        self.features_metadata, self.clip_index_metadata = features_metadata, clip_index_metadata
        self.reasoning_metadata, self.default_t0_us = reasoning_metadata, default_t0_us
        self.cache_base_dir = Path(cache_base_dir)
        self.shuffle_seed = int(shuffle_seed)
        self.epoch = 0
        self.distributed_rank = 0
        self.distributed_world_size = 1
        self._active_state: dict[str, int] | None = None
        self._resume_state: dict[str, int] | None = None
        self._metadata_avdi = PhysicalAIAVDatasetLocalInterface(
            local_dir=self.local_dir,
            chunk_ids=chunk_ids,
            features_metadata=features_metadata,
            clip_index_metadata=clip_index_metadata,
            reasoning_metadata=reasoning_metadata,
            split=split,
        )
        self.split = split
        self.chunk_ids = self._metadata_avdi.get_all_chunk_ids()
        self._sample_count_by_chunk = {chunk_id: 0 for chunk_id in self.chunk_ids}
        for clip_id, _ in self._metadata_avdi.get_sample_anchors(default_t0_us):
            chunk_id = self._metadata_avdi.get_clip_chunk(clip_id)
            if chunk_id in self._sample_count_by_chunk:
                self._sample_count_by_chunk[chunk_id] += 1
        kwargs = dict(
            num_history_steps=num_history_steps,
            num_future_steps=num_future_steps,
            time_step_s=time_step,
            num_frames=num_frames,
        )
        if camera_features is not None:
            kwargs["camera_features"] = tuple(camera_features)
        self.contract = PAIDataContract(**kwargs)
        self.include_extr_intr = include_extr_intr
        self.reshape_tensors_for_rl = reshape_tensors_for_rl
        self.validate_samples = validate_samples
        if preprocessor is not None and vla_preprocess_args is not None:
            raise ValueError("Specify either preprocessor or vla_preprocess_args, not both")
        self.preprocessor = preprocessor
        if vla_preprocess_args is not None:
            self.preprocessor = _instantiate_preprocessor(vla_preprocess_args, model_config)

    @property
    def clip_ids(self) -> list[str]:
        return self._metadata_avdi.get_all_clip_ids()

    def __len__(self) -> int:
        return self._samples_per_rank()

    def _rank_chunks(self, rank: int | None = None) -> list[int]:
        selected_rank = self.distributed_rank if rank is None else rank
        rank_chunks = self.chunk_ids[selected_rank :: self.distributed_world_size]
        return self._shuffle_chunks(rank_chunks, rank=selected_rank)

    def _shuffle_chunks(self, chunks: list[int], rank: int = 0) -> list[int]:
        if len(chunks) < 2:
            return list(chunks)
        generator = torch.Generator()
        generator.manual_seed(self.shuffle_seed + self.epoch + rank * 1_000_003)
        order = torch.randperm(len(chunks), generator=generator).tolist()
        return [chunks[index] for index in order]

    def _chunk_sample_count(self, chunks: list[int]) -> int:
        return sum(self._sample_count_by_chunk[chunk_id] for chunk_id in chunks)

    def _samples_per_rank(self) -> int:
        return min(
            self._chunk_sample_count(self._rank_chunks(rank))
            for rank in range(self.distributed_world_size)
        )

    def _get_worker_chunks(self) -> list[int]:
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        rank_chunks = self._rank_chunks()
        return rank_chunks[worker_id::num_workers]

    @staticmethod
    def _allocate_worker_quotas(capacities: list[int], total: int) -> list[int]:
        quotas = [0] * len(capacities)
        remaining = total
        while remaining > 0:
            active = [index for index, capacity in enumerate(capacities) if quotas[index] < capacity]
            if not active:
                raise RuntimeError("worker chunk capacities cannot satisfy the rank sample limit")
            share = max(1, (remaining + len(active) - 1) // len(active))
            for index in active:
                added = min(share, capacities[index] - quotas[index], remaining)
                quotas[index] += added
                remaining -= added
                if remaining == 0:
                    break
        return quotas

    def _worker_sample_limit(self) -> int:
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        rank_chunks = self._rank_chunks()
        worker_chunks = [rank_chunks[index::num_workers] for index in range(num_workers)]
        capacities = [self._chunk_sample_count(chunks) for chunks in worker_chunks]
        return self._allocate_worker_quotas(capacities, self._samples_per_rank())[worker_id]

    def set_distributed(self, rank: int, world_size: int) -> None:
        rank = int(rank)
        world_size = int(world_size)
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("distributed rank/world_size must satisfy 0 <= rank < world_size")
        self.distributed_rank = rank
        self.distributed_world_size = world_size

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic new chunk permutation for an epoch boundary."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch != self.epoch:
            self.epoch = epoch
            self._active_state = None
            self._resume_state = None

    def state_dict(self) -> dict[str, int]:
        """Return this worker's next unread chunk/anchor position for torchdata."""
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        state = self._active_state or {
            "next_chunk_position": 0,
            "next_anchor_position": 0,
            "remaining": self._worker_sample_limit(),
        }
        return {
            "version": 1,
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "distributed_rank": self.distributed_rank,
            "distributed_world_size": self.distributed_world_size,
            "worker_id": worker_id,
            "num_workers": num_workers,
            **state,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore a worker state saved by ``StatefulDataLoader``."""
        required = {
            "version", "shuffle_seed", "epoch", "distributed_rank", "distributed_world_size",
            "worker_id", "num_workers", "next_chunk_position", "next_anchor_position", "remaining",
        }
        missing = required.difference(state_dict)
        if missing:
            raise ValueError(f"Incomplete IterablePAIDataset state: missing {sorted(missing)}")
        if state_dict["version"] != 1 or state_dict["shuffle_seed"] != self.shuffle_seed:
            raise ValueError("IterablePAIDataset state is incompatible with the configured shuffle seed")
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        expected = (self.distributed_rank, self.distributed_world_size, worker_id, num_workers)
        actual = tuple(state_dict[key] for key in (
            "distributed_rank", "distributed_world_size", "worker_id", "num_workers"
        ))
        if actual != expected:
            raise ValueError(
                "IterablePAIDataset resume requires unchanged distributed rank/world size and DataLoader workers"
            )
        self.epoch = int(state_dict["epoch"])
        self._resume_state = {
            key: int(state_dict[key])
            for key in ("next_chunk_position", "next_anchor_position", "remaining")
        }
        self._active_state = dict(self._resume_state)

    def _worker_cache_dir(self) -> Path:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        if self.distributed_rank == 0:
            directory = f"{type(self).__name__}_{worker_id}"
        else:
            directory = f"{type(self).__name__}_rank{self.distributed_rank}_{worker_id}"
        path = self.cache_base_dir / directory
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _ensure_metadata_links(self, cache_dir: Path) -> None:
        for relative in (self.features_metadata, self.clip_index_metadata, "metadata"):
            source, target = self.local_dir / relative, cache_dir / relative
            if target.is_symlink() and not target.exists():
                target.unlink()
            if target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())
        if self.reasoning_metadata is not None:
            configured = Path(self.reasoning_metadata)
            source = configured if configured.is_absolute() else self.local_dir / configured
            target = cache_dir / (Path(source.name) if configured.is_absolute() else configured)
            if not target.exists() and not target.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source)

    def _feature_pairs(self) -> list[tuple[str, str]]:
        pairs = [("labels", "egomotion")]
        pairs += [("camera", feature.rsplit("/", 1)[-1]) for feature in self.contract.camera_features]
        if self.include_extr_intr:
            pairs += [
                ("calibration", "camera_intrinsics"),
                ("calibration", "sensor_extrinsics"),
                ("calibration", "vehicle_dimensions"),
            ]
        return pairs

    def pai_chunk_download(self, chunk_index: int, timeout_seconds=600, max_retries=3) -> Path:
        cache_dir = self._worker_cache_dir()
        self._ensure_metadata_links(cache_dir)
        from huggingface_hub import snapshot_download
        chunk = f"chunk_{int(chunk_index):04d}"
        patterns = [
            f"{component}/{part}/{part}.{chunk}.*" for component, part in self._feature_pairs()
        ]
        for attempt in range(max_retries):
            # Cleanup any leftover state from previous attempts to avoid conflicts
            if attempt > 0:
                logger.info("Cleaning up leftover state to prepare for attempt %d...", attempt + 1)
                self._cleanup_cache_dir(cache_dir, chunk_index=chunk_index)
                shutil.rmtree(cache_dir / ".cache" , ignore_errors=True) # Clear the previous temporary lock

            # Use a ThreadPoolExecutor to run the snapshot_download in a separate thread, allowing us to enforce a timeout
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(
                    snapshot_download,
                    repo_id=DEFAULT_REPO_ID,
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                    allow_patterns=list(dict.fromkeys(patterns)),
                    max_workers=1,
                )
                # Wait for the result, or timeout
                future.result(timeout=timeout_seconds)
                break
                
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Timeout while downloading chunk %d (attempt %d/%d), retrying...",
                    chunk_index, attempt + 1, max_retries
                )
                
            except Exception as e:
                logger.warning(
                    "Error while downloading chunk %d (attempt %d/%d): %s, retrying...",
                    chunk_index, attempt + 1, max_retries, str(e)
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                
        else:
            raise TimeoutError(f"Failed to download chunk {chunk_index} after {max_retries} attempts")

        # Clean up the isolated cache directory
        shutil.rmtree(cache_dir / ".cache", ignore_errors=True)
        return cache_dir

    def pai_chunk_delete(self, chunk_index: int) -> None:
        cache_dir = self._worker_cache_dir()
        self._cleanup_cache_dir(cache_dir, chunk_index=chunk_index)

    def _cleanup_cache_dir(self, cache_dir: Path, chunk_index: int | None = None) -> None:
        chunk = "chunk_*" if chunk_index is None else f"chunk_{int(chunk_index):04d}"
        for component, part in self._feature_pairs():
            parent = cache_dir / component / part
            if parent.is_dir():
                for path in parent.glob(f"{part}.{chunk}.*"):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

    def cleanup_cache(self) -> None:
        """Remove feature payloads while preserving metadata links and HF bookkeeping."""
        pattern = f"{type(self).__name__}_*"
        for cache_dir in self.cache_base_dir.glob(pattern):
            if cache_dir.is_dir():
                self._cleanup_cache_dir(cache_dir)

    def _load_stream_anchor(self, avdi: Any, clip_id: str, t0_us: int) -> dict[str, Any]:
        sample = load_physical_aiavdataset(
            clip_id, t0_us, avdi, self.contract, validate=self.validate_samples
        )
        if self.include_extr_intr:
            sample["extr"] = avdi.get_clip_feature(clip_id, "sensor_extrinsics")
            sample["intr"] = avdi.get_clip_feature(clip_id, "camera_intrinsics")
            dimensions = avdi.get_clip_feature(clip_id, "vehicle_dimensions")
            length = float(dimensions.length)
            if length <= 0:
                raise ValueError(f"Clip {clip_id!r} has a non-positive vehicle length")
            sample["ego_lwh"] = torch.tensor(
                [dimensions.length, dimensions.width, dimensions.height], dtype=torch.float32
            )
            sample["ego_length_offset"] = torch.tensor(
                dimensions.rear_axle_to_bbox_center / length, dtype=torch.float32
            )
        if getattr(avdi, "reasoning_db", None) is not None:
            sample.update(avdi.get_reasoning_data(clip_id, t0_us))
        if self.reshape_tensors_for_rl:
            PAIDataset._reshape_for_rl(sample)
        if self.preprocessor is not None:
            sample["tokenized_data"] = self.preprocessor(data=sample)
        return sample

    def _process_and_cleanup_chunk(
        self, chunk_index: int, start_anchor_position: int = 0
    ) -> Iterator[dict[str, Any]]:
        cache_dir = self.pai_chunk_download(chunk_index)
        try:
            avdi = PhysicalAIAVDatasetLocalInterface(
                local_dir=cache_dir,
                chunk_ids=[chunk_index],
                features_metadata=self.features_metadata,
                clip_index_metadata=self.clip_index_metadata,
                reasoning_metadata=self.reasoning_metadata,
                split=self.split,
            )
            for anchor_position, (clip_id, t0_us) in enumerate(
                avdi.get_sample_anchors(self.default_t0_us)
            ):
                if anchor_position < start_anchor_position:
                    continue
                yield self._load_stream_anchor(avdi, clip_id, t0_us)
        finally:
            self.pai_chunk_delete(chunk_index)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # Recover bounded storage after a previous worker/job was interrupted.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self._cleanup_cache_dir(self._worker_cache_dir())
        worker_chunks = self._get_worker_chunks()
        state = self._resume_state
        self._resume_state = None
        if state is None:
            state = {
                "next_chunk_position": 0,
                "next_anchor_position": 0,
                "remaining": self._worker_sample_limit(),
            }
        chunk_position = state["next_chunk_position"]
        anchor_position = state["next_anchor_position"]
        remaining = state["remaining"]
        if chunk_position < 0 or anchor_position < 0 or remaining < 0:
            raise ValueError("IterablePAIDataset resume positions must be non-negative")
        if chunk_position > len(worker_chunks):
            raise ValueError("IterablePAIDataset resume chunk position exceeds this worker's chunk list")
        self._active_state = dict(state)
        if remaining == 0:
            return
        for current_chunk_position in range(chunk_position, len(worker_chunks)):
            chunk_index = worker_chunks[current_chunk_position]
            start_anchor_position = anchor_position if current_chunk_position == chunk_position else 0
            self._active_state.update(
                next_chunk_position=current_chunk_position,
                next_anchor_position=start_anchor_position,
                remaining=remaining,
            )
            iterator = self._process_and_cleanup_chunk(chunk_index, start_anchor_position)
            try:
                for current_anchor_position, sample in enumerate(iterator, start=start_anchor_position):
                    remaining -= 1
                    if current_anchor_position + 1 >= self._sample_count_by_chunk[chunk_index]:
                        next_state = (current_chunk_position + 1, 0)
                    else:
                        next_state = (current_chunk_position, current_anchor_position + 1)
                    self._active_state.update(
                        next_chunk_position=next_state[0],
                        next_anchor_position=next_state[1],
                        remaining=remaining,
                    )
                    yield sample
                    if remaining == 0:
                        return
            finally:
                iterator.close()
