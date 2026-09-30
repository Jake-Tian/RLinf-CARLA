#!/usr/bin/env python3
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

"""Collect SFT data by driving CARLA with CARLA's own expert.

Frames come from the same ``CarlaEnvWorker`` the rollout uses, so both share one
distribution; routes are traced through the real map, not ``straight_route``.

  python3 collect_sft_data.py --server-dir DIR --out DATA --routes DATA/routes
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time

import numpy as np

from rlinf.envs.sim.carla.reward import RewardConfig
from rlinf.envs.sim.carla.server import ServerConfig
from rlinf.envs.sim.carla.worker import CarlaEnvWorker, CarlaWorkerConfig

#: Weather presets, by name so the module stays importable without ``carla``. No night.
WEATHER_NAMES = (
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "WetCloudyNoon",
    "SoftRainNoon",
    "MidRainyNoon",
    "ClearSunset",
    "CloudySunset",
    "WetSunset",
)

#: Straight-line separation a route needs before it is worth driving; below this
#: the episode ends before the policy has seen anything.
MIN_ROUTE_M = 90.0

#: How many spawn-point pairs to try before giving up on finding a long one.
PAIR_ATTEMPTS = 60


def import_agents(server_dir: str):
    """BehaviorAgent and GlobalRoutePlanner out of the CARLA tree.

    Not pip-installable: they live in the server distribution under
    PythonAPI/carla and import ``carla`` themselves. The path goes on sys.path
    and the import happens here, not at module scope, so this file still
    imports with no CARLA present.
    """
    api = os.path.join(server_dir, "PythonAPI", "carla")
    if not os.path.isdir(api):
        raise RuntimeError(
            f"no PythonAPI/carla under {server_dir!r}; the agents module ships "
            f"with the server distribution, not with the carla wheel"
        )
    if api not in sys.path:
        sys.path.insert(0, api)
    from agents.navigation.behavior_agent import BehaviorAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    return BehaviorAgent, GlobalRoutePlanner


def pick_route_pair(spawn_points, rng) -> tuple[int, int] | None:
    """A (start, end) spawn-point index pair far enough apart to be worth it."""
    n = len(spawn_points)
    if n < 2:
        return None
    for _ in range(PAIR_ATTEMPTS):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        a, b = spawn_points[i].location, spawn_points[j].location
        if math.dist((a.x, a.y), (b.x, b.y)) >= MIN_ROUTE_M:
            return i, j
    return None


def save_route(path: str, trace) -> int:
    """Write a GlobalRoutePlanner trace as ``x y`` lines. Returns the count.

    The format is what ``Route.from_file`` reads, and writing it by hand here
    rather than importing the writer keeps this script independent of the env
    package's internals.
    """
    pts = []
    for item in trace:
        # trace_route returns (Waypoint, RoadOption) pairs; older revisions
        # return bare Waypoints. Accept both rather than pin a version.
        wp = item[0] if isinstance(item, (tuple, list)) else item
        loc = wp.transform.location
        pts.append((loc.x, loc.y))
    # Consecutive duplicates make Route.project divide by a zero-length
    # segment; it skips them, but they bloat the file and the progress sum.
    deduped = [
        p for k, p in enumerate(pts) if k == 0 or math.dist(p, pts[k - 1]) > 1e-6
    ]
    if len(deduped) < 2:
        raise RuntimeError("traced a route with fewer than two distinct points")
    with open(path, "w") as f:
        f.write(f"# CARLA GlobalRoutePlanner, sampling point count {len(deduped)}\n")
        for x, y in deduped:
            f.write(f"{x:.3f} {y:.3f}\n")
    return len(deduped)


def spawn_traffic(
    world, carla_map, client, blueprint_lib, tm_port: int, count: int, rng
) -> list:
    """Put background vehicles on autopilot. Returns them for teardown.

    Traffic Manager must be in synchronous mode whenever the world is, or it
    keeps stepping at its own rate and the vehicles teleport between frames.

    Spawn points come from a *fresh* ``list(map.get_spawn_points())``, never the
    caller's list: the shuffle below is in place, and ``main`` still indexes its
    own list by ``end_idx`` to set the agent's destination, so shuffling the
    caller's would silently point the agent at a different spawn point than the
    route it was given. Nothing would raise.
    """
    if count <= 0:
        return []
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    vehicles = []
    car_bps = blueprint_lib.filter("vehicle.*")
    spawn_points = list(carla_map.get_spawn_points())
    rng.shuffle(spawn_points)
    for sp in spawn_points[:count]:
        bp = car_bps[int(rng.integers(len(car_bps)))]
        try:
            actor = world.try_spawn_actor(bp, sp)
        except RuntimeError:
            # A blueprint that will not spawn at this point is not an error
            # worth stopping for; one fewer background car changes nothing.
            continue
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        vehicles.append(actor)
    return vehicles


def despawn_traffic(client, tm_port: int, vehicles) -> None:
    """Remove the background vehicles and stop the traffic manager driving them.

    Order is the whole point. The traffic manager is shut down *before* any of
    its vehicles is destroyed: while it is still running it holds its own
    references to the actors it manages, and destroying one out from under it
    aborts the client process at the C++ level -- not a Python exception, so no
    ``try``/``except`` in the caller can catch it.

    Each destroy is still guarded individually: one vehicle that the server
    has already removed must not strand the rest.
    """
    try:
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(False)
        tm.shut_down()
    except Exception:  # noqa: BLE001 - teardown must not mask the run
        pass
    for actor in vehicles:
        try:
            actor.destroy()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/carla_cache"))
    ap.add_argument("--out", required=True, help="dataset directory")
    ap.add_argument(
        "--routes", default=None, help="route directory (default: <out>/routes)"
    )
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--port", type=int, default=2100)
    ap.add_argument("--tm-port", type=int, default=8100)
    ap.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="pin the renderer via DRI_PRIME. Only 0 and 1 work.",
    )
    ap.add_argument("--quality-level", default="Epic")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--num-routes", type=int, default=80)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument(
        "--max-steps", type=int, default=500, help="episode truncation horizon"
    )
    ap.add_argument(
        "--traffic", type=int, default=15, help="background autopilot vehicles"
    )
    ap.add_argument("--no-weather-random", action="store_true")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    import carla

    # Before the server: this import pulls CARLA's agents package and its own deps
    # (shapely, networkx), and a failure here is far cheaper than after Unreal has booted.
    BehaviorAgent, GlobalRoutePlanner = import_agents(args.server_dir)

    routes_dir = args.routes or os.path.join(args.out, "routes")
    logdir = args.logdir or os.path.join(args.out, "logs")
    images_dir = os.path.join(args.out, "images")
    for d in (routes_dir, logdir, images_dir):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    server_cfg = ServerConfig(
        server_dir=args.server_dir,
        cache_dir=args.cache_dir,
        base_port=args.port,
        quality_level=args.quality_level,
        startup_timeout=args.timeout,
        gpu=args.gpu,
    )
    worker_cfg = CarlaWorkerConfig(
        server=server_cfg,
        reward=RewardConfig(mode="sparse"),
        fps=args.fps,
        image_width=args.width,
        image_height=args.height,
        action_mode="control",
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )

    worker = CarlaEnvWorker(worker_cfg, 0, logdir)
    print(f"starting server on port {args.port} (gpu={args.gpu})", flush=True)
    worker.start()
    world = worker._world
    carla_map = worker._map
    blueprint_lib = world.get_blueprint_library()

    # Built once. Constructing the topology graph costs tens of seconds on a
    # full map; trace_route against a built planner is milliseconds.
    print("building route planner topology ...", flush=True)
    grp = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

    # -- routes ------------------------------------------------------------
    spawn_points = list(worker._spawn_points)
    if len(spawn_points) < 2:
        print("FATAL: map has fewer than two spawn points")
        return 2

    written = []
    for k in range(args.num_routes):
        pair = pick_route_pair(spawn_points, rng)
        if pair is None:
            print(
                f"  route {k}: no spawn pair at least {MIN_ROUTE_M:.0f} m "
                f"apart after {PAIR_ATTEMPTS} attempts; stopping at "
                f"{len(written)} routes"
            )
            break
        i, j = pair
        start = spawn_points[i].location
        end = carla_map.get_waypoint(spawn_points[j].location).transform.location
        trace = grp.trace_route(start, end)
        path = os.path.join(routes_dir, f"route_{k:04d}_from{i}_to{j}.txt")
        try:
            n_pts = save_route(path, trace)
        except RuntimeError as exc:
            print(f"  route {k}: {exc}; skipped")
            continue
        written.append((path, i, j, n_pts))
    print(f"routes: {len(written)} written to {routes_dir}", flush=True)
    if not written:
        print("FATAL: no usable routes; nothing to collect")
        return 2

    traffic = spawn_traffic(
        world, carla_map, worker._client, blueprint_lib, args.tm_port, args.traffic, rng
    )
    print(
        f"traffic: {len(traffic)} autopilot vehicles on tm port {args.tm_port}",
        flush=True,
    )

    # -- collection --------------------------------------------------------
    index_path = os.path.join(args.out, "index.jsonl")
    episodes = []
    total = 0
    t_start = time.time()

    with open(index_path, "w") as index_f:
        for ep in range(args.episodes):
            path, start_idx, end_idx, _ = written[int(rng.integers(len(written)))]
            weather_name = (
                WEATHER_NAMES[int(rng.integers(len(WEATHER_NAMES)))]
                if not args.no_weather_random
                else None
            )
            if weather_name:
                world.set_weather(getattr(carla.WeatherParameters, weather_name))

            # spawn_index too: reset() picks its own spawn point, which would start off-route.
            worker.cfg = dataclasses.replace(
                worker_cfg, route_file=path, spawn_index=start_idx
            )
            try:
                obs = worker.reset()
            except RuntimeError as exc:
                print(f"  ep {ep:03d}: reset failed: {exc}")
                continue

            agent = BehaviorAgent(worker._ego, behavior="normal")
            agent.set_destination(spawn_points[end_idx].location)

            ep_dir = os.path.join(images_dir, f"ep{ep:04d}")
            os.makedirs(ep_dir, exist_ok=True)

            reason = "max_steps"
            frames = 0
            for t in range(args.max_steps):
                frame_before = world.get_snapshot().frame
                ctrl = agent.run_step()
                # run_step must not advance the world, or every action would pair with the
                # wrong image.
                if world.get_snapshot().frame != frame_before:
                    raise RuntimeError(
                        "BehaviorAgent.run_step() advanced the world; the "
                        "recorded action/image pairing would be off by a tick"
                    )

                action = [float(ctrl.steer), float(ctrl.throttle), float(ctrl.brake)]
                obs, reward, term, trunc, info = worker.step(action)
                frames += 1

                rel = os.path.relpath(os.path.join(ep_dir, f"{t:05d}.jpg"), args.out)
                # Straight from the worker, channel order included: see _write_jpeg.
                frame = obs["main_images"]
                _write_jpeg(os.path.join(args.out, rel), frame, args.jpeg_quality)

                index_f.write(
                    json.dumps(
                        {
                            "image": rel,
                            "action": action,
                            "states": [float(x) for x in obs["states"]],
                            "task": obs["task_descriptions"],
                            "episode": ep,
                            "step": t,
                            "route": os.path.relpath(path, args.out),
                            "weather": weather_name,
                            "route_completion": float(info["route_completion"]),
                        }
                    )
                    + "\n"
                )

                if term or trunc or agent.done():
                    reason = info["outcome"] if (term or trunc) else "destination"
                    break

            episodes.append(
                {
                    "episode": ep,
                    "route": os.path.relpath(path, args.out),
                    "weather": weather_name,
                    "frames": frames,
                    "reason": reason,
                    "traffic": len(traffic),
                }
            )
            total += frames
            print(
                f"  ep {ep:03d}: {frames:4d} frames  {reason:12s} "
                f"{weather_name or 'default'}  total={total}",
                flush=True,
            )

    # Before teardown: a crash there must not discard the summary of a dataset already on disk.
    meta = {
        "server_dir": args.server_dir,
        "quality_level": args.quality_level,
        "fps": args.fps,
        "image_width": args.width,
        "image_height": args.height,
        "camera_fov": worker_cfg.camera_fov,
        "camera_transform": list(worker_cfg.camera_transform),
        "max_steps": args.max_steps,
        "traffic": args.traffic,
        "weather_random": not args.no_weather_random,
        "seed": args.seed,
        "routes": len(written),
        "episodes": episodes,
        "total_frames": total,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    outcomes = {}
    for e in episodes:
        outcomes[e["reason"]] = outcomes.get(e["reason"], 0) + 1
    print()
    print(f"frames total : {total}")
    print(f"episodes     : {len(episodes)}  {outcomes}")
    print(
        f"elapsed      : {meta['elapsed_s']:.0f}s "
        f"({total / max(meta['elapsed_s'], 1e-9):.1f} frames/s)"
    )
    print(f"dataset      : {args.out}")
    print(f"COLLECT_DONE frames={total} episodes={len(episodes)}")

    # Last, after everything worth keeping is written; see despawn_traffic for the ordering.
    despawn_traffic(worker._client, args.tm_port, traffic)
    worker.close()
    return 0


def _write_jpeg(path: str, frame: np.ndarray, quality: int) -> None:
    """Write one observation to disk, byte-for-byte as the worker produced it.

    JPEG rather than PNG because the raw frame is 2.7 MB and 30k of those is
    80 GB, where q90 lands around 150 KB each.

    Channel order: **the worker returns BGR, not RGB.** CARLA delivers
    ``raw_data`` as BGRA and ``_sync_image`` slices to the first three channels,
    so channel 0 is blue. Nothing is swapped here: reading the file back with
    ``np.array(Image.open(p))`` reproduces the worker's array bit-for-bit, so
    the policy sees at SFT time what it sees at rollout time. Swapping to make
    the JPEG look correct would break that parity, silently.
    """
    from PIL import Image

    Image.fromarray(frame).save(path, format="JPEG", quality=quality)


if __name__ == "__main__":
    raise SystemExit(main())
