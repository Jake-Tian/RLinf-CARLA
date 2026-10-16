"""Use observed optimizer counters rather than the adapter's expected batch count."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("metrics", type=Path)
args = parser.parse_args()
metrics = json.loads(args.metrics.read_text())
root = Path(metrics["source"])
ages = [json.loads(line) for line in (root / "optimizer_age_audit.jsonl").read_text().splitlines()]
publications = [json.loads(line) for line in (root / "publication_optimizer_audit.jsonl").read_text().splitlines()]
begins = {int(row["actor_version"]) + 1: int(row["optimizer_steps_before_training"]) for row in ages}
published = {int(row["version"]): row for row in publications}
counts = {}
differences = {}
for step, row in metrics["raw_scalars_by_step"].items():
    step = int(step)
    if step + 1 in begins:
        end = begins[step + 1]
    else:
        publication = published[step]
        assert publication["stable_during_sync"], "Final publication has uncertain counter bounds"
        end = int(publication["optimizer_steps_at_completion"])
    count = end - begins[step]
    assert count >= 0
    counts[str(step)] = count
    expected = row["train/actor/optimizer_steps_per_rank"]
    if abs(count - expected) > 1e-3:
        differences[str(step)] = {"adapter_expected": expected, "observed_actual": count}
metrics["actual_optimizer_counts_by_step"] = counts
metrics["optimizer_counter_audit"] = {
    "method": "difference of actual pre-training counters; final stable publication supplies final end counter; DP ranks are not summed",
    "total_actual_updates": sum(counts.values()),
    "differences_from_adapter_expected": differences,
}
steps = metrics["steady_runner_steps"]
updates = sum(counts[str(step)] for step in steps)
seconds = sum(metrics["raw_scalars_by_step"][str(step)]["time/step"] for step in steps)
metrics["steady"]["optimizer_updates"] = updates
metrics["steady"]["actual_optimizer_updates_per_second"] = updates / seconds
args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(metrics["optimizer_counter_audit"], indent=2))
