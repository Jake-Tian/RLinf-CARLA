# SPDX-License-Identifier: Apache-2.0
"""Frozen temporal, camera, and geometry contract for PAI samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}

DEFAULT_CAMERA_FEATURES = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)


@dataclass(frozen=True)
class PAIDataContract:
    """Configuration shared by SFT, rollout, reward, and evaluation.

    Timestamps are computed with integer microseconds to avoid floating-point
    ``arange`` changing a horizon by one sample.
    """

    num_history_steps: int = 16
    num_future_steps: int = 64
    time_step_s: float = 0.1
    camera_features: tuple[str, ...] = DEFAULT_CAMERA_FEATURES
    num_frames: int = 4

    def __post_init__(self) -> None:
        if self.num_history_steps < 1 or self.num_future_steps < 1 or self.num_frames < 1:
            raise ValueError("history, future, and image frame counts must be positive")
        if not np.isfinite(self.time_step_s) or self.time_step_s <= 0:
            raise ValueError("time_step_s must be finite and positive")
        unknown = set(self.camera_features) - CAMERA_NAME_TO_INDEX.keys()
        if unknown:
            raise ValueError(f"Unknown camera features: {sorted(unknown)}")
        if len(set(self.camera_features)) != len(self.camera_features):
            raise ValueError("camera_features must be unique")

    @property
    def time_step_us(self) -> int:
        value = self.time_step_s * 1_000_000
        rounded = round(value)
        if not np.isclose(value, rounded):
            raise ValueError("time_step_s must be representable as an integer number of microseconds")
        return int(rounded)

    @property
    def camera_indices(self) -> tuple[int, ...]:
        return tuple(sorted(CAMERA_NAME_TO_INDEX[name] for name in self.camera_features))

    def history_timestamps(self, t0_us: int) -> np.ndarray:
        offsets = np.arange(-(self.num_history_steps - 1), 1, dtype=np.int64)
        return np.int64(t0_us) + offsets * self.time_step_us

    def future_timestamps(self, t0_us: int) -> np.ndarray:
        offsets = np.arange(1, self.num_future_steps + 1, dtype=np.int64)
        return np.int64(t0_us) + offsets * self.time_step_us

    def image_timestamps(self, t0_us: int) -> np.ndarray:
        offsets = np.arange(-(self.num_frames - 1), 1, dtype=np.int64)
        return np.int64(t0_us) + offsets * self.time_step_us

    def validate_t0(self, t0_us: int) -> None:
        if isinstance(t0_us, bool) or not isinstance(t0_us, (int, np.integer)):
            raise TypeError("t0_us must be an integer timestamp in microseconds")
        if self.history_timestamps(int(t0_us))[0] < 0:
            raise ValueError("t0_us does not leave enough history inside the relative clip timeline")


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


def validate_pai_sample(sample: dict[str, Any], contract: PAIDataContract) -> None:
    """Fail fast when a loaded sample violates the frozen geometry contract."""

    required = {
        "clip_id",
        "t0_us",
        "image_frames",
        "camera_indices",
        "absolute_timestamps",
        "relative_timestamps",
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
        "ego_future_rot",
    }
    missing = sorted(required - sample.keys())
    if missing:
        raise ValueError(f"PAI sample is missing required fields: {missing}")

    n_cam = len(contract.camera_features)
    expected = {
        "camera_indices": (n_cam,),
        "absolute_timestamps": (n_cam, contract.num_frames),
        "relative_timestamps": (n_cam, contract.num_frames),
        "ego_history_xyz": (contract.num_history_steps, 3),
        "ego_history_rot": (contract.num_history_steps, 3, 3),
        "ego_future_xyz": (contract.num_future_steps, 3),
        "ego_future_rot": (contract.num_future_steps, 3, 3),
    }
    for key, expected_shape in expected.items():
        if _shape(sample[key]) != expected_shape:
            raise ValueError(f"{key} has shape {_shape(sample[key])}, expected {expected_shape}")

    images = sample["image_frames"]
    if images.ndim != 5 or images.shape[:2] != (n_cam, contract.num_frames) or images.shape[2] != 3:
        raise ValueError(
            "image_frames must have shape "
            f"({n_cam}, {contract.num_frames}, 3, H, W), got {_shape(images)}"
        )

    camera_indices = torch.as_tensor(sample["camera_indices"], dtype=torch.int64)
    if tuple(camera_indices.tolist()) != contract.camera_indices:
        raise ValueError(
            f"camera_indices must be sorted as {contract.camera_indices}, got {tuple(camera_indices.tolist())}"
        )

    absolute_timestamps = torch.as_tensor(sample["absolute_timestamps"], dtype=torch.int64)
    if torch.any(torch.diff(absolute_timestamps, dim=1) < 0):
        raise ValueError("camera timestamps must be monotonic within each camera")

    for key in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot"):
        if not torch.isfinite(torch.as_tensor(sample[key])).all():
            raise ValueError(f"{key} contains NaN or Inf")

    history_xyz = torch.as_tensor(sample["ego_history_xyz"])
    history_rot = torch.as_tensor(sample["ego_history_rot"])
    if not torch.allclose(history_xyz[-1], torch.zeros(3, dtype=history_xyz.dtype), atol=1e-4):
        raise ValueError("ego_history_xyz[-1] is not the t0 ego-frame origin")
    if not torch.allclose(history_rot[-1], torch.eye(3, dtype=history_rot.dtype), atol=1e-4):
        raise ValueError("ego_history_rot[-1] is not the t0 ego-frame identity")

