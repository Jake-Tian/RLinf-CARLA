"""Read complete TensorBoard scalar records without importing TensorFlow."""

import argparse
import csv
import json
import statistics
import struct
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tensorboard.compat.proto.event_pb2 import Event


def collect(root, actor_world_size, gpu_count, gpu_roles=None, warmup_rounds=1):
    steps = {}
    completion_times = {}
    for path in sorted((root / "tensorboard").glob("events.out.tfevents.*")):
        with path.open("rb") as stream:
            while True:
                header = stream.read(12)
                if len(header) != 12:
                    break
                length = struct.unpack("<Q", header[:8])[0]
                payload = stream.read(length)
                footer = stream.read(4)
                if len(payload) != length or len(footer) != 4:
                    break
                event = Event.FromString(payload)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        row = steps.setdefault(event.step, {})
                        row[value.tag] = value.simple_value
                        if value.tag == "time/step":
                            completion_times[event.step] = event.wall_time
    completed = {step: row for step, row in steps.items() if "time/step" in row}
    ordered_steps = sorted(completed)
    steady = {step: completed[step] for step in ordered_steps[warmup_rounds:]}
    result = {"source": str(root), "runner_iterations_logged": len(completed),
              "steady_runner_steps": sorted(steady), "raw_scalars_by_step": completed,
              "completion_timestamp_by_step": completion_times}
    if steady:
        durations = [row["time/step"] for row in steady.values()]
        seconds = sum(durations)
        def total_samples(tag):
            totals = [row[tag] * actor_world_size for row in steady.values()]
            for total in totals:
                if abs(total - round(total)) > 1e-3:
                    raise ValueError(f"Nonintegral reconstructed sample count: {total}")
            return sum(round(total) for total in totals)
        valid = total_samples("train/actor/valid_samples_per_rank")
        new = total_samples("train/actor/new_samples_per_rank")
        updates = sum(row["train/actor/optimizer_steps_per_rank"] for row in steady.values())
        result["steady"] = {
            "iteration_mean_s": statistics.mean(durations),
            "iteration_std_s": statistics.stdev(durations) if len(durations) > 1 else 0,
            "new_chunks_per_second": new / seconds,
            "valid_chunks_per_second": valid / seconds,
            "actual_optimizer_updates_per_second": updates / seconds,
            "gpu_seconds_per_valid_chunk": gpu_count * seconds / valid,
            "new_chunks": new, "valid_chunks": valid, "optimizer_updates": updates,
            "phase_mean_s": {tag: statistics.mean(row[tag] for row in steady.values() if tag in row)
                             for tag in ("time/actor_training", "time/rollout/predict",
                                         "time/rollout/generate_one_epoch", "time/env/env_interact_step",
                                         "time/actor/wait_for_rollout_store_ready")
                             if any(tag in row for row in steady.values())},
        }
        if warmup_rounds and ordered_steps[warmup_rounds-1] in completion_times and (root / "gpu_util.csv").exists() and (gpu_roles or (gpu_count == 8 and actor_world_size == 3)):
            begin = completion_times[ordered_steps[warmup_rounds-1]]
            end = completion_times[max(steady)]
            samples = {}
            with (root / "gpu_util.csv").open(newline="") as stream:
                for row in csv.reader(stream):
                    if len(row) == 5:
                        timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                        timestamp = timestamp.replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
                        samples.setdefault(int(row[1]), []).append((timestamp, float(row[3]), float(row[4])))
            roles = gpu_roles or {"env": (0, 1, 2), "actor": (3, 4, 5), "rollout": (6, 7)}
            role_metrics = {}
            for role, ids in roles.items():
                duration = util_time = power_time = zero_time = 0.0
                for gpu in ids:
                    series = sorted(samples.get(gpu, []))
                    for current, following in zip(series, series[1:]):
                        dt = max(0.0, min(end, following[0]) - max(begin, current[0]))
                        duration += dt
                        util_time += current[1] * dt
                        power_time += current[2] * dt
                        zero_time += dt if current[1] == 0 else 0.0
                if duration:
                    util = util_time / duration
                    role_metrics[role] = {"gpu_count": len(ids), "physical_gpu_indices": list(ids), "mean_util_pct": util,
                                          "mean_power_w": power_time / duration,
                                          "zero_util_sample_time_pct": 100 * zero_time / duration,
                                          "idle_equivalent_gpu_s_per_iteration": len(ids) * statistics.mean(durations) * (1 - util / 100)}
            result["gpu_steady"] = {"window_start_epoch": begin, "window_end_epoch": end,
                                     "window_definition": f"exclude first {warmup_rounds} completed rounds; integrate through final completed round",
                                     "roles": role_metrics}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--actor-world-size", type=int, default=3)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-roles", type=json.loads)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    args = parser.parse_args()
    result = collect(args.root, args.actor_world_size, args.gpu_count, args.gpu_roles, args.warmup_rounds)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(json.dumps({key: value for key, value in result.items() if key != "raw_scalars_by_step"}, indent=2))
