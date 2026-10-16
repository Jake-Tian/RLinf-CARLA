# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy local-filesystem access to the Physical AI AV dataset layout."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLIP_RELATIVE_DURATION_US = 20_000_000


def parse_chunk_ids(chunk_ids: str | int | Iterable[int] | None) -> tuple[int, ...] | None:
    """Normalize chunk selectors; ranges use the recipe's end-exclusive semantics."""

    if chunk_ids is None:
        return None
    if isinstance(chunk_ids, bool):
        raise TypeError("chunk_ids cannot be boolean")
    if isinstance(chunk_ids, int):
        return (chunk_ids,)
    if isinstance(chunk_ids, str):
        parts = [part.strip() for part in chunk_ids.split("-")]
        if len(parts) != 2 or not all(parts):
            raise ValueError("chunk range must have the end-exclusive form 'START-END'")
        start, end = map(int, parts)
        if start < 0 or end <= start:
            raise ValueError("chunk range must satisfy 0 <= START < END")
        return tuple(range(start, end))
    if isinstance(chunk_ids, Iterable):
        values = tuple(int(value) for value in chunk_ids)
        if any(value < 0 for value in values):
            raise ValueError("chunk ids must be non-negative")
        return values
    raise TypeError(f"Unsupported chunk_ids type: {type(chunk_ids).__name__}")


class PAIFeatureCatalog:
    """Minimal replacement for ``physical_ai_av.dataset.Features`` path lookup."""

    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df

    def get_chunk_feature_filename(self, chunk_id: int, feature: str) -> str:
        return str(self.features_df.at[feature, "chunk_path"]).format(chunk_id=int(chunk_id))

    def get_clip_files_in_zip(self, clip_id: str, feature: str) -> dict[str, str]:
        templates = self.features_df.at[feature, "clip_files_in_zip"]
        if not isinstance(templates, dict):
            raise ValueError(f"Feature {feature!r} has no clip_files_in_zip mapping")
        return {key: value.format(clip_id=clip_id) for key, value in templates.items()}


class PhysicalAIAVDatasetLocalInterface:
    """Index PAI metadata eagerly and open large feature payloads lazily."""

    def __init__(
        self,
        local_dir: str | Path,
        chunk_ids: str | int | Iterable[int] | None = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        reasoning_metadata: str | None = None,
        split: str | None = None,
        start_safe_margin_seconds: float = 1.6,
        end_safe_margin_seconds: float = 6.4,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.chunk_ids = parse_chunk_ids(chunk_ids)
        self.start_safe_margin_seconds = start_safe_margin_seconds
        self.end_safe_margin_seconds = end_safe_margin_seconds

        features_df = pd.read_csv(self.local_dir / features_metadata, index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            lambda value: json.loads(value) if isinstance(value, str) else value
        )
        self.features = PAIFeatureCatalog(features_df)
        self.clip_index = pd.read_parquet(self.local_dir / clip_index_metadata)
        if not self.clip_index.index.is_unique:
            raise ValueError("clip index must contain unique clip_id values")
        if split is not None:
            if "split" not in self.clip_index.columns:
                raise ValueError("clip index metadata must contain a 'split' column when split is set")
            self.clip_index = self.clip_index.loc[self.clip_index["split"] == split].copy()

        self.reasoning_db: dict[str, dict[str, Any]] | None = None
        if reasoning_metadata is not None:
            reasoning_path = Path(reasoning_metadata)
            if not reasoning_path.is_absolute():
                reasoning_path = self.local_dir / reasoning_path
            if not reasoning_path.is_file():
                raise FileNotFoundError(f"Reasoning metadata does not exist: {reasoning_path}")
            self.reasoning_db = self._read_reasoning_data(pd.read_parquet(reasoning_path))
            self._filter_clips_by_reasoning_events()

        presence_path = self.local_dir / "metadata/feature_presence.parquet"
        self.sensor_presence = pd.read_parquet(presence_path) if presence_path.is_file() else None

    @staticmethod
    def _read_reasoning_data(reasoning_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if "events" not in reasoning_df.columns:
            raise ValueError("reasoning metadata must contain an 'events' column")
        result: dict[str, dict[str, Any]] = {}
        for clip_id, events_cell in reasoning_df["events"].items():
            if isinstance(events_cell, str):
                events = json.loads(events_cell) if events_cell.strip() else []
            elif events_cell is None or (np.isscalar(events_cell) and pd.isna(events_cell)):
                events = []
            else:
                events = list(events_cell)
            timestamps: list[int] = []
            cots: list[str] = []
            for event in events:
                if isinstance(event, dict) and "event_start_timestamp" in event:
                    timestamps.append(int(event["event_start_timestamp"]))
                    cots.append(str(event.get("cot", "")))
            result[str(clip_id)] = {
                "event_t0s": np.asarray(timestamps, dtype=np.int64),
                "cot": cots,
            }
        return result

    def _filter_clips_by_reasoning_events(self) -> None:
        assert self.reasoning_db is not None
        start_us = round(self.start_safe_margin_seconds * 1_000_000)
        end_us = round(self.end_safe_margin_seconds * 1_000_000)
        kept: dict[str, dict[str, Any]] = {}
        for clip_id, entry in self.reasoning_db.items():
            timestamps = entry["event_t0s"]
            mask = (timestamps >= start_us) & (timestamps + end_us <= CLIP_RELATIVE_DURATION_US)
            indices = np.flatnonzero(mask)
            if len(indices):
                kept[clip_id] = {
                    "event_t0s": timestamps[mask],
                    "cot": [entry["cot"][index] for index in indices],
                }
        self.reasoning_db = kept
        self.clip_index = self.clip_index.loc[self.clip_index.index.astype(str).isin(kept)].copy()

    def get_all_clip_ids(self) -> list[str]:
        index = self.clip_index
        if self.chunk_ids is not None:
            index = index.loc[index["chunk"].isin(self.chunk_ids)]
        return index.index.astype(str).tolist()

    def get_all_chunk_ids(self) -> list[int]:
        """Return sorted chunk ids represented by the selected clip index."""
        if "chunk" not in self.clip_index.columns:
            raise ValueError("clip index metadata must contain a 'chunk' column")
        index = self.clip_index
        if self.chunk_ids is not None:
            index = index.loc[index["chunk"].isin(self.chunk_ids)]
        return sorted({int(value) for value in index["chunk"].dropna().tolist()})

    def get_sample_anchors(self, default_t0_us: int | None) -> list[tuple[str, int]]:
        anchors: list[tuple[str, int]] = []
        for clip_id in self.get_all_clip_ids():
            if self.reasoning_db is not None:
                anchors.extend((clip_id, int(t0)) for t0 in self.reasoning_db[clip_id]["event_t0s"])
            elif default_t0_us is not None:
                anchors.append((clip_id, int(default_t0_us)))
            else:
                raise ValueError("default_t0_us is required when no reasoning metadata provides event anchors")
        return anchors

    def get_clip_chunk(self, clip_id: str) -> int:
        return int(self.clip_index.at[clip_id, "chunk"])

    def get_reasoning_data(self, clip_id: str, t0_us: int) -> dict[str, str]:
        if self.reasoning_db is None or clip_id not in self.reasoning_db:
            raise KeyError(f"No reasoning data for clip {clip_id!r}")
        matches = np.flatnonzero(self.reasoning_db[clip_id]["event_t0s"] == int(t0_us))
        if len(matches) != 1:
            raise KeyError(f"No unique reasoning event for ({clip_id!r}, {t0_us})")
        return {"cot": self.reasoning_db[clip_id]["cot"][int(matches[0])]}

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        del maybe_stream  # This implementation is intentionally local-only.
        if feature not in self.features.features_df.index:
            raise KeyError(f"Unknown PAI feature {feature!r}")
        chunk_path = self.local_dir / self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        if not chunk_path.is_file():
            raise FileNotFoundError(f"PAI feature payload is not downloaded: {chunk_path}")
        if chunk_path.suffix == ".parquet":
            return pd.read_parquet(chunk_path).loc[clip_id]
        if chunk_path.suffix != ".zip":
            raise ValueError(f"Unsupported PAI feature payload: {chunk_path}")

        members = self.features.get_clip_files_in_zip(clip_id, feature)
        with ZipFile(chunk_path) as archive:
            if feature == "egomotion":
                try:
                    from physical_ai_av.egomotion import EgomotionState
                except ImportError as exc:
                    raise ImportError(
                        "physical_ai_av is required to decode PAI egomotion payloads"
                    ) from exc
                frame = pd.read_parquet(io.BytesIO(archive.read(members["egomotion"])))
                return EgomotionState.from_egomotion_df(frame).create_interpolator(
                    frame["timestamp"].to_numpy()
                )
            if feature.startswith("camera"):
                try:
                    from physical_ai_av.video import SeekVideoReader
                except ImportError as exc:
                    raise ImportError("physical_ai_av is required to decode PAI camera payloads") from exc
                timestamps = pd.read_parquet(io.BytesIO(archive.read(members["frame_timestamps"])))
                return SeekVideoReader(
                    video_data=io.BytesIO(archive.read(members["video"])),
                    timestamps=timestamps["timestamp"].to_numpy(),
                )
            return {
                key: pd.read_parquet(io.BytesIO(archive.read(member)))
                if member.endswith(".parquet")
                else io.BytesIO(archive.read(member))
                for key, member in members.items()
            }
