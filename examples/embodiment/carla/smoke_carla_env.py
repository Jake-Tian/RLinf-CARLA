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

"""First end-to-end run of the CARLA env against a live server.

Constant action, no trainer or model; a blank frame fails the run instead of looking healthy.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

#: Gentle throttle, no steering: advances route progress without leaving the road.
ACTION = [0.0, 0.5, 0.0]


def _as_numpy(x):
    """CarlaEnv returns torch tensors; convert once here rather than at each use."""
    return x.numpy() if hasattr(x, "numpy") else x


def gpu_memory_mib(index: int) -> int | None:
    """VRAM used on one GPU, in MiB. None if it cannot be read."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--id=%d" % index,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - telemetry must never kill the run
        return None


def import_env_cls():
    """Return the CARLA env registered in RLinf."""
    from rlinf.envs.sim.carla import CarlaEnv

    return CarlaEnv, "rlinf.envs.sim.carla"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument(
        "--steps", type=int, default=200, help="timed steps, after the reset"
    )
    ap.add_argument(
        "--server-dir",
        default=os.environ.get("CARLA_SERVER_DIR", "/path/to/CARLA_0916"),
    )
    ap.add_argument("--base-port", type=int, default=2100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--quality-level", default="Epic")
    ap.add_argument("--image-width", type=int, default=1280)
    ap.add_argument("--image-height", type=int, default=720)
    ap.add_argument("--reward-mode", default="sparse")
    ap.add_argument("--max-episode-steps", type=int, default=1200)
    ap.add_argument("--logdir", default="carla_smoke_logs")
    ap.add_argument(
        "--json-out", default=None, help="write the metrics dict here as JSON"
    )
    args = ap.parse_args()

    if not os.path.isdir(args.server_dir):
        print(f"FATAL: no CARLA server at {args.server_dir}")
        print("  (0.9.16 is unpacked only after its transfer completes)")
        return 2

    import numpy as np

    CarlaEnv, where = import_env_cls()
    print(f"CarlaEnv from {where}")

    cfg = {
        "server_dir": args.server_dir,
        "base_port": args.base_port,
        "gpu": args.gpu,
        "quality_level": args.quality_level,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "reward_mode": args.reward_mode,
        "max_episode_steps": args.max_episode_steps,
        "auto_reset": True,
        "ignore_terminations": False,
    }

    metrics: dict = {
        "num_envs": args.num_envs,
        "steps": args.steps,
        "quality_level": args.quality_level,
        "resolution": [args.image_width, args.image_height],
    }
    vram_before = gpu_memory_mib(args.gpu)

    # --- construct + start: server launch is inside this call --------------
    t0 = time.perf_counter()
    try:
        env = CarlaEnv(
            cfg,
            args.num_envs,
            seed_offset=0,
            total_num_processes=args.num_envs,
            worker_info={"gpu": args.gpu},
            logdir=args.logdir,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL during construction/startup: {type(exc).__name__}: {exc}")
        print("  server logs are under", os.path.abspath(args.logdir))
        return 1
    startup_s = time.perf_counter() - t0
    metrics["startup_s"] = startup_s
    metrics["vram_before_mib"] = vram_before
    print(f"started {args.num_envs} env(s) in {startup_s:.1f} s")

    try:
        # --- reset --------------------------------------------------------
        t0 = time.perf_counter()
        try:
            obs, _ = env.reset()
        except Exception as exc:  # noqa: BLE001
            print(f"\nFAIL on reset: {type(exc).__name__}: {exc}")
            return 1
        reset_s = time.perf_counter() - t0
        metrics["reset_s"] = reset_s
        print(
            f"reset in {reset_s:.2f} s  "
            f"main_images={tuple(obs['main_images'].shape)} "
            f"states={tuple(obs['states'].shape)}"
        )

        # A non-zero, non-constant frame means the off-screen renderer produced an image.
        frame = _as_numpy(obs["main_images"])
        fmin, fmax, fmean = float(frame.min()), float(frame.max()), float(frame.mean())
        metrics.update(frame_min=fmin, frame_max=fmax, frame_mean=fmean)
        metrics["frame_is_rendering"] = bool(fmax > 0 and fmean > 1.0)
        print(
            f"frame pixel min={fmin:.1f} max={fmax:.1f} mean={fmean:.2f}"
            f"  -> rendering={'YES' if metrics['frame_is_rendering'] else 'NO'}"
        )
        if not metrics["frame_is_rendering"]:
            print("\nFAIL: frames are blank. The server is alive but rendering")
            print("  nothing -- the RenderOffScreen failure this probe exists for.")
            return 1

        vram_after = gpu_memory_mib(args.gpu)
        metrics["vram_after_mib"] = vram_after
        if vram_before is not None and vram_after is not None:
            metrics["vram_delta_mib"] = vram_after - vram_before
            print(
                f"VRAM on gpu {args.gpu}: {vram_before} -> {vram_after} MiB "
                f"(+{vram_after - vram_before})"
            )

        # --- timed loop ---------------------------------------------------
        actions = np.tile(np.array(ACTION, dtype=np.float32), (args.num_envs, 1))
        times: list[float] = []
        episodes_done = 0
        for i in range(args.steps):
            t = time.perf_counter()
            obs, reward, term, trunc, infos = env.step(actions)
            times.append(time.perf_counter() - t)
            episodes_done += int(_as_numpy(term | trunc).sum())
            if (i + 1) % 50 == 0:
                print(
                    f"  step {i + 1:5d}  "
                    f"last={times[-1] * 1000:7.1f} ms  "
                    f"mean={statistics.fmean(times) * 1000:7.1f} ms  "
                    f"episodes={episodes_done}"
                )

        steps_s = len(times) / sum(times)
        metrics.update(
            step_ms_mean=statistics.fmean(times) * 1000,
            step_ms_median=statistics.median(times) * 1000,
            step_ms_min=min(times) * 1000,
            step_ms_max=max(times) * 1000,
            step_ms_p95=sorted(times)[int(0.95 * len(times)) - 1] * 1000,
            steps_per_s=steps_s,
            # The number that drives training: env steps per wall second over the pool.
            env_steps_per_s=steps_s * args.num_envs,
            episodes_completed=episodes_done,
        )
        try:
            metrics["outcomes"] = env.outcome_summary()
        except Exception:  # noqa: BLE001
            pass

        print("\n=== throughput ===")
        for k in (
            "step_ms_mean",
            "step_ms_median",
            "step_ms_min",
            "step_ms_max",
            "step_ms_p95",
        ):
            print(f"  {k:16s} {metrics[k]:8.1f} ms")
        print(f"  {'steps/s':16s} {metrics['steps_per_s']:8.2f}")
        print(
            f"  {'env-steps/s':16s} {metrics['env_steps_per_s']:8.2f}"
            f"   <- across {args.num_envs} env(s)"
        )
        print(f"  episodes completed: {episodes_done}")
        if "outcomes" in metrics:
            print(f"  outcomes: {metrics['outcomes']}")

    finally:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001
            print(f"warning: close() raised {type(exc).__name__}: {exc}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nmetrics -> {args.json_out}")
    print("\nSMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
