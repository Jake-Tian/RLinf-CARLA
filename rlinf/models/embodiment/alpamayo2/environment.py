# Copyright 2026 The RLinf-CARLA Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Real multi-camera observations and feedback control for Alpamayo2."""

import dataclasses
import math
import os
import queue
from collections import deque

import numpy as np

from rlinf.envs.sim.carla.worker import CarlaEnvWorker

from .geometry import acceleration_control, relative_history


class AlpamayoCarlaWorker(CarlaEnvWorker):
    """Preserve CARLA lifecycle and reward, adding the checkpoint's observations."""

    def __init__(self, cfg, worker_index, logdir):
        cache = os.environ.get(
            "CARLA_CACHE_DIR", os.path.join(os.path.abspath(logdir), "cache")
        )
        cfg = dataclasses.replace(
            cfg,
            camera_fov=120.0,
            server=dataclasses.replace(cfg.server, cache_dir=cache),
        )
        super().__init__(cfg, worker_index, logdir)
        self.repeat = round(cfg.fps / 10.0)
        if self.repeat < 1 or not math.isclose(cfg.fps, self.repeat * 10.0):
            raise ValueError("Alpamayo2 requires a CARLA rate divisible by 10 Hz")
        self._extra_cameras = []
        self._extra_queues = []
        self._poses = deque(maxlen=16)
        self._frames = deque(maxlen=4)
        self._ticks = 0
        self._target_speed = 0.0
        self._wheelbase = 2.8
        self._max_steer = math.radians(35.0)

    def _destroy_actors(self):
        for camera in self._extra_cameras:
            try:
                camera.destroy()
            except RuntimeError:
                pass
        self._extra_cameras.clear()
        self._extra_queues.clear()
        self._poses.clear()
        self._frames.clear()
        super()._destroy_actors()

    def reset(self):
        import carla

        super().reset()
        library = self._world.get_blueprint_library()
        for yaw, fov in ((-55.0, 120.0), (55.0, 120.0), (0.0, 30.0)):
            blueprint = library.find("sensor.camera.rgb")
            for key, value in (
                ("image_size_x", self.cfg.image_width),
                ("image_size_y", self.cfg.image_height),
                ("fov", fov),
            ):
                blueprint.set_attribute(key, str(value))
            cx, cy, cz = self.cfg.camera_transform
            camera = self._world.spawn_actor(
                blueprint,
                carla.Transform(
                    carla.Location(x=cx, y=cy, z=cz), carla.Rotation(yaw=yaw)
                ),
                attach_to=self._ego,
            )
            images = queue.Queue()
            camera.listen(images.put)
            self._extra_cameras.append(camera)
            self._extra_queues.append(images)
        wheels = self._ego.get_physics_control().wheels
        self._wheelbase = (
            abs(
                np.mean([w.position.x for w in wheels[:2]])
                - np.mean([w.position.x for w in wheels[2:]])
            )
            / 100.0
        )
        self._max_steer = math.radians(max(w.max_steer_angle for w in wheels))
        self._ticks = 0
        initial_speed = float(os.environ.get("ALPAMAYO_INITIAL_SPEED", "5.0"))
        direction = self._ego.get_transform().get_forward_vector()
        self._ego.set_target_velocity(
            carla.Vector3D(
                x=direction.x * initial_speed, y=direction.y * initial_speed, z=0.0
            )
        )
        # Obtain real pose history before inference; no recorded future is supplied.
        self._ego.apply_control(carla.VehicleControl(throttle=0.12))
        for _ in range(16 * self.repeat):
            observation = self._obs(self._world.tick())
        self._target_speed = self._speed_mps()
        progress = self.route.progress(
            self._ego.get_location().x, self._ego.get_location().y
        )
        from rlinf.envs.sim.carla.reward import Outcome

        self.reward.step(progress, Outcome.RUNNING)
        self._eps_state.route_completion = progress
        return observation

    def _decode_action(self, action):
        controls, self._target_speed = acceleration_control(
            action,
            self._speed_mps(),
            self._target_speed,
            1.0 / self.cfg.fps,
            self._wheelbase,
            self._max_steer,
        )
        return controls

    def _read_rgb(self, images, frame):
        while True:
            image = images.get(timeout=self.cfg.frame_timeout_s)
            if image.frame >= frame:
                return (
                    np.frombuffer(image.raw_data, dtype=np.uint8)
                    .reshape(image.height, image.width, 4)[:, :, 2::-1]
                    .copy()
                )

    def _obs(self, frame):
        observation = super()._obs(frame)
        if not self._extra_cameras:
            return observation
        left, right, tele = [
            self._read_rgb(images, frame) for images in self._extra_queues
        ]
        self._ticks += 1
        if self._ticks % self.repeat == 0:
            front = observation["main_images"][:, :, ::-1].copy()
            self._frames.append(
                np.stack([left, front, right, tele]).transpose(0, 3, 1, 2)
            )
            self._poses.append(np.asarray(self._ego.get_transform().get_matrix()))
        if self._poses:
            poses = [self._poses[0]] * (16 - len(self._poses)) + list(self._poses)
            frames = [self._frames[0]] * (4 - len(self._frames)) + list(self._frames)
            xyz, rotation = relative_history(poses)
            observation.update(
                image_frames=np.stack(frames, axis=1),
                ego_history_xyz=xyz,
                ego_history_rot=rotation,
            )
        return observation
