# Copyright 2026 The RLinf-CARLA Contributors.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
repo = Path(__file__).resolve().parents[3]
rows = []
for rank in range(6):
    a = json.loads((root / f"full_first_update_rank{rank}.json").read_text())
    assert not a["missing_gradients"], a
    assert (
        math.isfinite(a["full_preclip_norm"])
        and abs(a["full_postclip_norm"] - 2) < 1e-4
    ), a
    assert a["optimizer_state_dtypes"] == {"torch.float32": 1234}, a
    for role in ["vision", "language"]:
        assert a["gradient_energy_by_role"][role] > 0, a
        assert math.isfinite(a["sampled_parameter_changes"][role]["max_abs_delta"]), a
    lp = [
        json.loads(s)
        for s in (root / f"initial_logprob_rank{rank}.jsonl").read_text().splitlines()
    ]
    assert lp and all(
        math.isfinite(x["max_abs"]) and x["max_abs"] <= 1e-6 for x in lp
    ), lp
    source = root / (
        f"actor_rank_{rank}_source.json"
        if mode == "async"
        else f"colocate_actor_rank{rank}_source.json"
    )
    origins = json.loads(source.read_text())
    for name, v in origins.items():
        path = Path(v["path"])
        assert path.is_relative_to(repo), (name, v)
        if name in ("model_compat", "processor"):
            assert path.is_relative_to(repo / "third_party/alpamayo"), (name, v)
    if mode == "async":
        name = (
            "publication_optimizer_audit.jsonl"
            if rank == 0
            else f"publication_optimizer_audit_rank{rank}.jsonl"
        )
        records = [json.loads(x) for x in (root / name).read_text().splitlines()]
        assert all(x["stable_during_sync"] for x in records), records
        assert records[-1]["optimizer_steps_at_completion"] == 4, records
    else:
        records = [
            json.loads(x)
            for x in (root / f"colocate_update_rank{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        assert records[-1]["optimizer_after"] == 4, records
    rows.append(
        {
            "rank": rank,
            "initial_logprob_max_abs": max(x["max_abs"] for x in lp),
            "passed": True,
            "sampled_parameter_changes": a["sampled_parameter_changes"],
        }
    )
for role in ["vision", "language"]:
    assert (
        max(v["sampled_parameter_changes"][role]["max_abs_delta"] for v in rows) > 0
    ), rows
initial = [json.loads(f.read_text()) for f in root.glob("policy_full_initial_*.json")]
assert initial and all(
    x["total_parameters"] == 2220993344 and x["trainable_parameters"] == 2220993344
    for x in initial
)
r = {
    "passed": True,
    "mode": mode,
    "actual_global_updates": 4,
    "ranks": rows,
    "standalone_import_guard": True,
    "scope": "Existing pinned runtime, bundled model source and blocked VeRL/StarVLA imports. Not a fresh OS installation or convergence test.",
}
(root / "acceptance_audit.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r), flush=True)
