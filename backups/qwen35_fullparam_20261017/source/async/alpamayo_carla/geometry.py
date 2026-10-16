"""Coordinate and control conversions for a right-handed driving policy."""
import math

import numpy as np


def relative_history(transforms):
    """Convert CARLA world poses to the current ego frame with y pointing left."""
    reflection = np.diag([1.0, -1.0, 1.0, 1.0])
    poses = reflection @ np.asarray(transforms, dtype=np.float64) @ reflection
    current = poses[-1]
    rotation = current[:3, :3]
    xyz = (poses[:, :3, 3] - current[:3, 3]) @ rotation
    rotations = rotation.T @ poses[:, :3, :3]
    return xyz.astype(np.float32), rotations.astype(np.float32)


def acceleration_control(action, speed, target_speed, dt, wheelbase, max_steer):
    """Track acceleration and left-positive curvature using CARLA controls."""
    acceleration, curvature = np.asarray(action, dtype=float).reshape(2)
    if not np.isfinite([acceleration, curvature]).all():
        raise ValueError("acceleration and curvature must be finite")
    target_speed = max(0.0, target_speed + acceleration * dt)
    error = target_speed - speed
    throttle = np.clip(0.8 * error + max(acceleration, 0.0) / 3.0 + 0.12, 0, 1)
    brake = np.clip(-0.8 * error + max(-acceleration, 0.0) / 6.0, 0, 1)
    if brake > 0.05:
        throttle = 0.0
    else:
        brake = 0.0
    steer = np.clip(math.atan(-curvature * wheelbase) / max_steer, -1, 1)
    return (float(steer), float(throttle), float(brake)), target_speed
