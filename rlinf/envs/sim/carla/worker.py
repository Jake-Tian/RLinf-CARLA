# Copyright 2026 The RLinf-CARLA Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One CARLA environment: one server process, one client connection, one ego."""

from __future__ import annotations

import math
import os
import queue
from dataclasses import dataclass, field

import numpy as np

from .reward import CarlaReward, EpisodeState, Outcome, RewardConfig
from .route import Route, straight_route
from .server import CarlaServer, ServerConfig

#: CARLA splits collisions by what was hit; kept though the reward collapses them.
_COLLISION_KINDS = {"walker": "collisions_pedestrian", "vehicle": "collisions_vehicle"}

#: Components of the ``states`` vector, in order; must line up with the model side.
STATE_FIELDS = (
    "speed_mps",
    "accel_long",
    "accel_lat",
    "yaw_rate",
    "steer",
    "throttle",
    "brake",
    "route_offset_m",
    "heading_error_rad",
)


@dataclass
class CarlaWorkerConfig:
    server: ServerConfig
    reward: RewardConfig = field(default_factory=RewardConfig)

    host: str = "127.0.0.1"
    #: Simulation rate; 20 Hz is what the Bench2Drive numbers were produced at.
    fps: float = 20.0

    image_width: int = 1280
    image_height: int = 720
    camera_fov: float = 90.0
    #: Camera mount, roughly a windscreen position on a sedan.
    camera_transform: tuple[float, float, float] = (1.5, 0.0, 2.4)

    #: "control" -> [steer, throttle, brake]; "waypoints" is not implemented.
    action_mode: str = "control"

    max_episode_steps: int = 1200
    warmup_ticks: int = 20
    #: How long to wait for a tick's frame; a false positive here kills the episode.
    frame_timeout_s: float = 10.0

    #: Off-route threshold in metres; Bench2Drive publishes no value to copy.
    max_lateral_offset_m: float = 2.0

    #: Route source; a file wins, otherwise a synthetic straight route.
    route_file: str | None = None
    route_length_m: float = 200.0

    seed: int = 0
    #: Distinct per worker, so two workers do not spawn on the same point.
    spawn_index: int | None = None

    def __post_init__(self) -> None:
        if self.action_mode not in ("control", "waypoints"):
            raise ValueError(f"unknown action_mode {self.action_mode!r}")
        if self.action_mode == "waypoints":
            raise NotImplementedError(
                "waypoint->control conversion is not implemented yet; "
                "the trajectory PID lives on the model side for now"
            )


class CarlaEnvWorker:
    """A single driving environment. Not thread-safe; one process each."""

    def __init__(self, cfg: CarlaWorkerConfig, worker_index: int, logdir: str) -> None:
        self.cfg = cfg
        self.worker_index = worker_index

        self.server = CarlaServer(cfg.server, worker_index, logdir)
        self.reward = CarlaReward(cfg.reward)

        self._client = None
        self._world = None
        self._map = None
        self._ego = None
        self._camera = None
        self._collision_sensor = None
        self._images: queue.Queue = queue.Queue()
        self._collisions: list[str] = []
        self._original_settings = None

        self.route: Route | None = None
        self._eps_state = EpisodeState()
        self._elapsed = 0
        self._rng = np.random.default_rng(cfg.seed + worker_index)
        self._spawn_points: list = []

        # Previous-step yaw, for the yaw-rate term.
        self._prev_yaw = None
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Bring up the server and connect; separate from __init__ so construction is cheap."""
        import carla

        self.server.start()
        ok, why = self.server.wait_ready()
        if not ok:
            raise RuntimeError(
                f"CARLA server for worker {self.worker_index} failed: {why}\n"
                f"{self.server.failure_report()}"
            )

        self._client = carla.Client(self.cfg.host, self.server.port)
        self._client.set_timeout(60.0)
        self._world = self._client.get_world()
        self._map = self._world.get_map()
        self._spawn_points = self._map.get_spawn_points()
        self._apply_sync_settings()

    def close(self) -> None:
        self._destroy_actors()
        self._restore_settings()
        self.server.stop()

    def __enter__(self) -> CarlaEnvWorker:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _apply_sync_settings(self) -> None:
        s = self._world.get_settings()
        self._original_settings = s
        s.synchronous_mode = True
        s.fixed_delta_seconds = 1.0 / self.cfg.fps
        # Must stay False; with rendering off the camera silently yields nothing.
        s.no_rendering_mode = False
        # Sync mode clamps fixed_delta_seconds to max_substep_delta_time * max_substeps.
        s.substepping = True
        s.max_substep_delta_time = 0.01
        s.max_substeps = max(10, math.ceil(s.fixed_delta_seconds / 0.01))
        self._world.apply_settings(s)

    def _restore_settings(self) -> None:
        if self._world is not None and self._original_settings is not None:
            try:
                self._world.apply_settings(self._original_settings)
            except Exception:  # noqa: BLE001 - teardown must not mask the real error
                pass

    def _destroy_actors(self) -> None:
        for actor in (self._collision_sensor, self._camera, self._ego):
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:  # noqa: BLE001
                    pass
        self._collision_sensor = self._camera = self._ego = None

    # -- episode -----------------------------------------------------------

    def _make_route(self, spawn=None) -> Route:
        if self.cfg.route_file:
            return Route.from_file(self._pick_route_file(self.cfg.route_file))
        # Synthetic fallback only; straight_route ignores the road, so RC never reaches 100.
        # `spawn` must be where the ego is placed, or the heading misses the car.
        if spawn is None:
            spawn = self._pick_spawn_point()
        yaw = math.radians(spawn.rotation.yaw)
        return straight_route(
            (spawn.location.x, spawn.location.y), yaw, self.cfg.route_length_m
        )

    def _spawn_at_route_start(self, route: Route):
        """The transform that places the ego on the route's first waypoint, z from the road."""
        import carla

        (x0, y0), (x1, y1) = route.waypoints[0], route.waypoints[1]
        waypoint = self._map.get_waypoint(carla.Location(x=x0, y=y0))
        if waypoint is None:
            raise RuntimeError(
                f"route starts at ({x0:.1f}, {y0:.1f}), which is not on any "
                f"road in this map; the route file and the map disagree"
            )
        return carla.Transform(
            waypoint.transform.location,
            carla.Rotation(yaw=math.degrees(math.atan2(y1 - y0, x1 - x0))),
        )

    def _pick_route_file(self, source: str) -> str:
        """Resolve ``route_file`` to one file; use a single file for GRPO, not a directory."""
        if os.path.isfile(source):
            return source
        if os.path.isdir(source):
            files = sorted(
                os.path.join(source, f)
                for f in os.listdir(source)
                if f.endswith(".txt")
            )
            if not files:
                raise RuntimeError(
                    f"route_file directory {source!r} holds no .txt routes; "
                    f"generate them before starting the env"
                )
            return files[int(self._rng.integers(len(files)))]
        raise RuntimeError(f"route_file {source!r} is neither a file nor a directory")

    def _pick_spawn_point(self) -> "object":
        if not self._spawn_points:
            raise RuntimeError("map has no spawn points")
        if self.cfg.spawn_index is not None:
            return self._spawn_points[self.cfg.spawn_index % len(self._spawn_points)]
        return self._spawn_points[int(self._rng.integers(len(self._spawn_points)))]

    def reset(self) -> dict:
        import carla

        self._destroy_actors()
        self._drain_images()
        self._collisions = []
        self._elapsed = 0
        self._eps_state = EpisodeState()
        self.reward.reset()
        self._prev_yaw = None
        self._prev_steer = self._prev_throttle = self._prev_brake = 0.0

        # Route first, then the spawn that belongs to it.
        if self.cfg.route_file:
            self.route = self._make_route()
            spawn = self._spawn_at_route_start(self.route)
        else:
            # Synthetic: exactly one spawn draw, and the route is built from it.
            spawn = self._pick_spawn_point()
            self.route = self._make_route(spawn=spawn)
        self.reward.set_route_steps(self.cfg.max_episode_steps)

        bp = self._world.get_blueprint_library()
        vehicle_bp = bp.filter("vehicle.*")[0]
        self._ego = self._world.try_spawn_actor(vehicle_bp, spawn)
        if self._ego is None:
            raise RuntimeError("could not spawn ego vehicle")

        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg.image_width))
        cam_bp.set_attribute("image_size_y", str(self.cfg.image_height))
        cam_bp.set_attribute("fov", str(self.cfg.camera_fov))
        cx, cy, cz = self.cfg.camera_transform
        self._camera = self._world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=cx, y=cy, z=cz)),
            attach_to=self._ego,
        )
        self._camera.listen(self._images.put)

        col_bp = bp.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego
        )
        self._collision_sensor.listen(self._on_collision)

        # Warmup: the first frames pay for texture streaming and shader compile.
        for _ in range(self.cfg.warmup_ticks):
            self._world.tick()
        # In sync mode a frame exists only on a tick, so return THIS tick's observation.
        frame = self._world.tick()
        return self._obs(frame)

    def _on_collision(self, event) -> None:
        other = event.other_actor
        kind = (
            "walker"
            if other is not None and "walker" in other.type_id
            else (
                "vehicle" if other is not None and "vehicle" in other.type_id else None
            )
        )
        self._collisions.append(_COLLISION_KINDS.get(kind, "collisions_layout"))

    def _drain_images(self) -> None:
        """Empty the sensor queue; otherwise the camera back-pressures the simulation."""
        while True:
            try:
                self._images.get_nowait()
            except queue.Empty:
                return

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Apply one control action, advance one tick, return (obs, reward, term, trunc, info)."""
        import carla

        steer, throttle, brake = self._decode_action(action)
        control = carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        )
        self._ego.apply_control(control)
        frame = self._world.tick()
        self._elapsed += 1

        self._prev_steer, self._prev_throttle, self._prev_brake = (
            steer,
            throttle,
            brake,
        )

        self._eps_state.infractions.update(self._collisions)
        self._collisions = []

        loc = self._ego.get_transform().location
        rc = self.route.progress(loc.x, loc.y) if self.route else 0.0
        self._eps_state.route_completion = rc

        speed = self._speed_mps()
        if speed < 0.1 and rc < 100.0:
            self._eps_state.infractions.add("min_speed_infractions")
        if self.route is not None and self.route.is_off_route(
            loc.x, loc.y, self.cfg.max_lateral_offset_m
        ):
            self._eps_state.infractions.add("outside_route_lanes")

        timed_out = self._elapsed >= self.cfg.max_episode_steps
        outcome = self.reward.classify(self._eps_state, timed_out)

        terminated = outcome in (Outcome.SUCCESS, Outcome.CRASH)
        truncated = timed_out and not terminated

        reward = self.reward.step(
            rc, outcome if (terminated or truncated) else Outcome.RUNNING
        )
        info = {
            "route_completion": rc,
            "outcome": outcome.value,
            "infractions": sorted(self._eps_state.infractions),
            "speed_mps": speed,
        }
        return self._obs(frame), reward, terminated, truncated, info

    def _decode_action(self, action) -> tuple[float, float, float]:
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < 3:
            raise ValueError(f"expected [steer, throttle, brake], got {a.size} values")
        # Clip rather than let CARLA clamp, so reward and physics agree on what happened.
        return (
            float(np.clip(a[0], -1.0, 1.0)),
            float(np.clip(a[1], 0.0, 1.0)),
            float(np.clip(a[2], 0.0, 1.0)),
        )

    # -- observation -------------------------------------------------------

    def _sync_image(self, frame: int) -> np.ndarray | None:
        """The frame rendered for tick ``frame``, or None if it never arrived; blocking."""
        img = None
        while True:
            try:
                candidate = self._images.get(timeout=self.cfg.frame_timeout_s)
            except queue.Empty:
                return None
            if candidate.frame >= frame:
                img = candidate
                break
            # Stale frame: rendered before the tick being observed; drop and wait.
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        return arr.reshape(img.height, img.width, 4)[:, :, :3].copy()

    def _speed_mps(self) -> float:
        v = self._ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def _ego_state_vector(self) -> np.ndarray:
        import carla  # noqa: F401 - imported for the Vector3D type in annotations

        tf = self._ego.get_transform()
        v = self._ego.get_velocity()
        acc = self._ego.get_acceleration()
        yaw = math.radians(tf.rotation.yaw)

        speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        # Rotate into the ego frame, so the vector is heading-independent.
        cy, sy = math.cos(-yaw), math.sin(-yaw)
        a_long = acc.x * cy - acc.y * sy
        a_lat = acc.x * sy + acc.y * cy

        yaw_rate = 0.0
        if self._prev_yaw is not None:
            dt = 1.0 / self.cfg.fps
            d = yaw - self._prev_yaw
            # Wrap to [-pi, pi]; crossing the seam would read as a violent spin.
            d = (d + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = d / dt
        self._prev_yaw = yaw

        loc = tf.location
        if self.route is not None:
            along, offset, _ = self.route.project(loc.x, loc.y)
            i = min(int(along), len(self.route.waypoints) - 1)
            j = min(i + 1, len(self.route.waypoints) - 1)
            rx = self.route.waypoints[j][0] - self.route.waypoints[i][0]
            ry = self.route.waypoints[j][1] - self.route.waypoints[i][1]
            route_heading = math.atan2(ry, rx)
        else:
            offset, route_heading = 0.0, yaw
        heading_error = (route_heading - yaw + math.pi) % (2 * math.pi) - math.pi

        return np.array(
            [
                speed,
                a_long,
                a_lat,
                yaw_rate,
                self._prev_steer,
                self._prev_throttle,
                self._prev_brake,
                offset,
                heading_error,
            ],
            dtype=np.float32,
        )

    def _task_description(self) -> str:
        """Navigation command rather than a language task."""
        if self.route is None:
            return "drive"
        return f"follow the route ({self.route.length_m:.0f} m), keep to the lane"

    def _obs(self, frame: int) -> dict:
        image = self._sync_image(frame)
        if image is None:
            # Not recoverable: the policy would get a zero image and learn from noise.
            raise RuntimeError(
                f"worker {self.worker_index}: no camera frame for tick {frame} "
                f"arrived within {self.cfg.frame_timeout_s:.0f}s "
                f"(episode step {self._elapsed}, "
                f"image_size={self.cfg.image_width}x{self.cfg.image_height})."
            )
        return {
            "main_images": image,
            "states": self._ego_state_vector(),
            "task_descriptions": self._task_description(),
            # wrist_images is deliberately absent; a placeholder would inflate the token count.
        }
