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

"""Restore the action-window keys a starVLA checkpoint's config.yaml dropped.

``baseframework.from_pretrained`` builds the policy from the run dir's
``config.yaml``, and ``MLP_ActionHeader.get_action_model`` reads
``future_action_window_size`` / ``past_action_window_size`` with no default, so a
missing key raises ``ConfigAttributeError`` rather than returning None. That file
is written by ``config_tracker.py``, which dumps only the keys the run *accessed*;
SFT reads ``action_horizon``, so the window keys never land.

The values come from ``config.full.yaml`` in the same run dir, never guessed, and
``past + 1 + future == action_horizon`` is asserted as a cross-check. A wrong
window does **not** trip ``load_state_dict(strict=True)`` -- chunk_len enters only
as a reshape -- so it loads cleanly and serves garbage.

    python3 repair_starvla_ckpt_config.py --run-dir <RUN_DIR> [--dry-run]
    python3 repair_starvla_ckpt_config.py --config <RUN_DIR>/config.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

FUTURE = "future_action_window_size"
PAST = "past_action_window_size"

#: Keys land in this order at the end of the block, which is also where they sort:
#: config_tracker emits the block alphabetically.
INDENT = "    "


def parse_simple_yaml_scalars(path: str) -> dict[str, object]:
    """Read the flat ``key: value`` pairs under ``framework.action_model``.

    Deliberately not a yaml parser: this runs before the venv is imported, and
    the block is a flat map of scalars. Quoted values stay strings.
    """
    out: dict[str, object] = {}
    with open(path) as f:
        lines = f.read().splitlines()
    in_framework = False
    in_action_model = False
    for line in lines:
        if line == "framework:":
            in_framework = True
            continue
        if in_framework and line.startswith("  ") and not line.startswith("    "):
            in_action_model = line.strip() == "action_model:"
            continue
        if in_action_model:
            if line.startswith("    "):
                key, _, value = line.strip().partition(":")
                value = value.strip().strip("'\"")
                try:
                    out[key] = int(value)
                except ValueError:
                    out[key] = value
            elif line.strip():
                break
    return out


def insert_window_keys(text: str, future: int, past: int) -> str:
    """Insert the two keys at the end of the ``framework.action_model`` block.

    Textual, not a yaml round-trip: the file is generated, and ``OmegaConf.save``
    would reformat unrelated parts of it (``version_id: '0.21'``, ``1.0e-06``).
    """
    lines = text.splitlines(keepends=True)

    start = None
    for i, line in enumerate(lines):
        if line.rstrip("\n") == "framework:":
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("  action_model:"):
                    start = j
                    break
            break
    if start is None:
        raise SystemExit(
            "! no 'framework:' -> '  action_model:' block. The config was not "
            "written by the starVLA tracker, or its shape changed. Refusing to "
            "guess -- edit it by hand."
        )

    # The block ends at the first line less indented than its members.
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.startswith(INDENT) or not line.strip():
            end += 1
            continue
        break

    new = [
        f"{INDENT}{FUTURE}: {future}\n",
        f"{INDENT}{PAST}: {past}\n",
    ]
    return "".join(lines[:end] + new + lines[end:])


def resolve_values(run_dir: str) -> tuple[int, int]:
    full = os.path.join(run_dir, "config.full.yaml")
    if not os.path.exists(full):
        raise SystemExit(
            f"! {full} is missing, so there is no record of what the run was "
            f"launched with. Deriving future from action_horizon would be a "
            f"guess about a reshape that fails silently -- supply the values by "
            f"hand instead."
        )
    vals = parse_simple_yaml_scalars(full)
    if FUTURE not in vals or PAST not in vals:
        raise SystemExit(f"! {full} has no {FUTURE}/{PAST} either.")
    future, past = int(vals[FUTURE]), int(vals[PAST])

    # The invariant that ties the window to what the data was chunked as.
    horizon = vals.get("action_horizon")
    if horizon is not None and past + 1 + future != int(horizon):
        raise SystemExit(
            f"! past+1+future = {past + 1 + future} != action_horizon = "
            f"{horizon} in {full}. One of the two files was edited; resolve "
            f"before loading, because the head would train/serve mismatched."
        )
    return future, past


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="starVLA run dir holding config.yaml")
    ap.add_argument("--config", help="config.yaml directly (overrides --run-dir)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.config:
        config_path = os.path.abspath(args.config)
        run_dir = os.path.dirname(config_path)
    elif args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        config_path = os.path.join(run_dir, "config.yaml")
    else:
        raise SystemExit("! pass --run-dir or --config")

    if not os.path.exists(config_path):
        raise SystemExit(f"! {config_path} does not exist")

    present = parse_simple_yaml_scalars(config_path)
    if FUTURE in present and PAST in present:
        print(
            f"already has {FUTURE}={present[FUTURE]} "
            f"{PAST}={present[PAST]} -- nothing to do"
        )
        return 0

    future, past = resolve_values(run_dir)

    with open(config_path) as f:
        text = f.read()
    patched = insert_window_keys(text, future, past)

    print(f"{config_path}")
    print(f"  adding {FUTURE}: {future}")
    print(f"  adding {PAST}: {past}")
    print(
        f"  action_horizon unchanged: {present.get('action_horizon')} "
        f"(past+1+future = {past + 1 + future})"
    )

    if args.dry_run:
        print("[dry-run] not written")
        return 0

    shutil.copy2(config_path, config_path + ".bak")
    with open(config_path, "w") as f:
        f.write(patched)

    # Reload through a real parser: "the text changed" is not "the loader finds them".
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        print(
            "! omegaconf unavailable; textual insert applied but unverified. "
            "Re-run under the starVLA venv to verify."
        )
        return 0
    cfg = OmegaConf.load(config_path)
    am = cfg.framework.action_model
    got_future = am.get(FUTURE)
    got_past = am.get(PAST)
    if got_future != future or got_past != past:
        _restore(config_path)
        raise SystemExit(
            f"! wrote the keys but reloading gives future={got_future} "
            f"past={got_past}. Backup restored; nothing changed."
        )
    chunk = int(got_past) + 1 + int(got_future)
    print(
        f"  verified by reload: future={got_future} past={got_past} "
        f"-> chunk_len={chunk}"
    )
    print(f"  backup: {config_path}.bak")
    return 0


def _restore(config_path: str) -> bool:
    bak = config_path + ".bak"
    if os.path.exists(bak):
        shutil.copy2(bak, config_path)
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
