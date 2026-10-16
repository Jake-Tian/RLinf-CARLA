# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert one ``(clip_id, t0_us)`` PAI anchor into model-ready tensors."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from autopilot.utils.dataset.pai_contract import CAMERA_NAME_TO_INDEX, PAIDataContract, validate_pai_sample


class PAIDataInterface(Protocol):
    """Small interface required by the sample loader."""

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any: ...


def _quaternion_xyzw_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Vectorized scipy-free conversion for normalized ``[x, y, z, w]`` quaternions."""

    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(~np.isfinite(q)) or np.any(norms == 0):
        raise ValueError("egomotion contains invalid quaternions")
    q = q / norms
    x, y, z, w = np.moveaxis(q, -1, 0)
    matrices = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[..., 0, 1] = 2 * (x * y - z * w)
    matrices[..., 0, 2] = 2 * (x * z + y * w)
    matrices[..., 1, 0] = 2 * (x * y + z * w)
    matrices[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[..., 1, 2] = 2 * (y * z - x * w)
    matrices[..., 2, 0] = 2 * (x * z - y * w)
    matrices[..., 2, 1] = 2 * (y * z + x * w)
    matrices[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _camera_name(feature: str) -> str:
    if not isinstance(feature, str):
        raise TypeError(f"camera feature must be a string, got {type(feature).__name__}")
    return feature.rsplit("/", maxsplit=1)[-1].lower()


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int,
    avdi: PAIDataInterface,
    contract: PAIDataContract | None = None,
    maybe_stream: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Load a PAI sample using an explicit, shared data contract.

    Unlike the upstream Alpamayo helper, this local-data adapter has no hidden
    network-backed interface and returns unbatched trajectories. Batching and
    trajectory-group dimensions belong to the later model adapter.
    """

    if not clip_id:
        raise ValueError("clip_id must be non-empty")
    contract = contract or PAIDataContract()
    contract.validate_t0(t0_us)
    t0_us = int(t0_us)

    egomotion = avdi.get_clip_feature(clip_id, "egomotion", maybe_stream=maybe_stream)
    if egomotion is None:
        raise ValueError(f"Clip {clip_id!r} has no egomotion feature")

    history = egomotion(contract.history_timestamps(t0_us))
    future = egomotion(contract.future_timestamps(t0_us))
    history_xyz = np.asarray(history.pose.translation, dtype=np.float64)
    future_xyz = np.asarray(future.pose.translation, dtype=np.float64)
    history_rot_world = _quaternion_xyzw_to_matrix(history.pose.rotation.as_quat())
    future_rot_world = _quaternion_xyzw_to_matrix(future.pose.rotation.as_quat())

    t0_xyz = history_xyz[-1]
    world_to_t0 = history_rot_world[-1].T
    history_xyz_local = (world_to_t0 @ (history_xyz - t0_xyz).T).T
    future_xyz_local = (world_to_t0 @ (future_xyz - t0_xyz).T).T
    history_rot_local = np.einsum("ij,tjk->tik", world_to_t0, history_rot_world)
    future_rot_local = np.einsum("ij,tjk->tik", world_to_t0, future_rot_world)

    vision = load_physical_aiavdataset_vision(
        clip_id, t0_us, avdi, contract, maybe_stream=maybe_stream
    )

    sample: dict[str, Any] = {
        "clip_id": clip_id,
        "t0_us": t0_us,
        **vision,
        "ego_history_xyz": torch.from_numpy(history_xyz_local).to(torch.float32),
        "ego_history_rot": torch.from_numpy(history_rot_local).to(torch.float32),
        "ego_future_xyz": torch.from_numpy(future_xyz_local).to(torch.float32),
        "ego_future_rot": torch.from_numpy(future_rot_local).to(torch.float32),
    }
    if validate:
        validate_pai_sample(sample, contract)
    return sample


def load_physical_aiavdataset_light(
    clip_id: str,
    t0_us: int,
    avdi: PAIDataInterface,
    contract: PAIDataContract | None = None,
    maybe_stream: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Load only egomotion-derived fields for RL prompt/reward construction."""
    if not clip_id:
        raise ValueError("clip_id must be non-empty")
    contract = contract or PAIDataContract()
    contract.validate_t0(int(t0_us))
    t0_us = int(t0_us)
    egomotion = avdi.get_clip_feature(clip_id, "egomotion", maybe_stream=maybe_stream)
    if egomotion is None:
        raise ValueError(f"Clip {clip_id!r} has no egomotion feature")
    history = egomotion(contract.history_timestamps(t0_us))
    future = egomotion(contract.future_timestamps(t0_us))
    history_xyz = np.asarray(history.pose.translation, dtype=np.float64)
    future_xyz = np.asarray(future.pose.translation, dtype=np.float64)
    history_rot_world = _quaternion_xyzw_to_matrix(history.pose.rotation.as_quat())
    future_rot_world = _quaternion_xyzw_to_matrix(future.pose.rotation.as_quat())
    t0_xyz = history_xyz[-1]
    world_to_t0 = history_rot_world[-1].T
    sample = {
        "clip_id": clip_id,
        "t0_us": t0_us,
        "ego_history_xyz": torch.from_numpy((world_to_t0 @ (history_xyz - t0_xyz).T).T).to(torch.float32),
        "ego_future_xyz": torch.from_numpy((world_to_t0 @ (future_xyz - t0_xyz).T).T).to(torch.float32),
        "ego_history_rot": torch.from_numpy(
            np.einsum("ij,tjk->tik", world_to_t0, history_rot_world)
        ).to(torch.float32),
        "ego_future_rot": torch.from_numpy(
            np.einsum("ij,tjk->tik", world_to_t0, future_rot_world)
        ).to(torch.float32),
    }
    if validate:
        # Validate only the geometry fields here; image fields are intentionally absent.
        for key in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot"):
            if not torch.isfinite(sample[key]).all():
                raise ValueError(f"{key} contains NaN or Inf")
    return sample


def load_physical_aiavdataset_vision(
    clip_id: str,
    t0_us: int,
    avdi: PAIDataInterface,
    contract: PAIDataContract | None = None,
    maybe_stream: bool = False,
) -> dict[str, Any]:
    """Load camera frames and timestamps without touching egomotion payloads."""
    contract = contract or PAIDataContract()
    contract.validate_t0(int(t0_us))
    requested_image_timestamps = contract.image_timestamps(int(t0_us))
    camera_data: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for feature in contract.camera_features:
        name = _camera_name(feature)
        camera = avdi.get_clip_feature(clip_id, feature, maybe_stream=maybe_stream)
        if camera is None:
            raise ValueError(f"Clip {clip_id!r} has no {feature!r} camera feature")
        frames, actual_timestamps = camera.decode_images_from_timestamps(requested_image_timestamps)
        frames = np.asarray(frames)
        actual_timestamps = np.asarray(actual_timestamps, dtype=np.int64)
        if frames.ndim != 4 or frames.shape[0] != contract.num_frames or frames.shape[-1] != 3:
            raise ValueError(f"Camera {feature!r} returned invalid frame shape {frames.shape}")
        if actual_timestamps.shape != (contract.num_frames,):
            raise ValueError(f"Camera {feature!r} returned invalid timestamp shape {actual_timestamps.shape}")
        camera_data.append(
            (
                CAMERA_NAME_TO_INDEX[name],
                torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2))),
                torch.from_numpy(actual_timestamps.copy()),
            )
        )
    camera_data.sort(key=lambda item: item[0])
    camera_indices = torch.tensor([item[0] for item in camera_data], dtype=torch.int64)
    absolute_timestamps = torch.stack([item[2] for item in camera_data])
    return {
        "image_frames": torch.stack([item[1] for item in camera_data]),
        "camera_indices": camera_indices,
        "requested_image_timestamps": torch.from_numpy(requested_image_timestamps.copy()),
        "absolute_timestamps": absolute_timestamps,
        "relative_timestamps": (absolute_timestamps - absolute_timestamps.min()).to(torch.float32) * 1e-6,
    }
