#!/usr/bin/env python
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

"""Preflight for the CARLA SFT dataset, before any GPU is spent on it.

Builds real samples through ``get_vla_dataset``, the call the trainer uses, and
checks the edges: a chunk straddling an episode still has the right shape.

    python check_sft_dataset.py [--config <yaml>] [--starvla <repo>]
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def find_starvla() -> str:
    return os.environ.get("STARVLA_PATH", "/path/to/starVLA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--starvla",
        default=find_starvla(),
        help="StarVLA-SFT checkout (put on sys.path for the imports below)",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="training yaml; defaults to this checkout's carla config",
    )
    ap.add_argument(
        "--data-config",
        default=None,
        help="the deployed data_config.py to load normalize/unnormalize from; "
        "defaults to the copy inside the StarVLA checkout",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=5,
        help="how many samples to build and check (the first, the last, and "
        "spread between)",
    )
    args = ap.parse_args()

    config = args.config or (
        f"{args.starvla}/examples/RLinfCARLA/CARLA/train_files/starvla_carla.yaml"
    )
    if args.data_config is None:
        args.data_config = (
            f"{args.starvla}/examples/RLinfCARLA/CARLA/train_files/"
            f"data_registry/data_config.py"
        )

    if args.starvla not in sys.path:
        sys.path.insert(0, args.starvla)

    failures = 0

    # -- 1. registry ------------------------------------------------------
    print("== registry ==")
    try:
        from starVLA.dataloader.gr00t_lerobot.registry import (
            DATASET_NAMED_MIXTURES,
            ROBOT_TYPE_CONFIG_MAP,
        )

        robots = [k for k in ROBOT_TYPE_CONFIG_MAP if "carla" in k]
        mixes = [k for k in DATASET_NAMED_MIXTURES if "carla" in k]
        print(f"  carla robot_types: {robots}")
        print(f"  carla mixtures   : {mixes}")
        print(f"  spec             : {DATASET_NAMED_MIXTURES.get('carla_sft')}")
        assert robots, (
            "no carla robot_type registered -- is data_config.py in "
            "examples/<x>/<y>/train_files/data_registry/?"
        )
        assert mixes, "no carla mixture registered"
    except Exception:
        failures += 1
        traceback.print_exc()

    # -- 2. one config, read once ----------------------------------------
    print("\n== config ==")
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config)
        vla = cfg.datasets.vla_data
        fw = cfg.framework.action_model
        horizon = int(fw.action_horizon)
        print(f"  action_dim={fw.action_dim} action_horizon={horizon}")
        print(
            f"  batch={vla.per_device_batch_size} obs_image_size={vla.obs_image_size}"
        )
        print(f"  data_mix={vla.data_mix} root={vla.data_root_dir}")
        # Two YAML sections disagreeing still trains, on a chunk nobody predicts.
        assert int(vla.get("action_horizon", horizon)) == horizon, (
            f"datasets.vla_data.action_horizon "
            f"{vla.get('action_horizon')} != framework.action_model."
            f"action_horizon {horizon}"
        )
        assert tuple(vla.obs_image_size) == tuple(vla.get("image_size", [])), (
            "obs_image_size and image_size differ; inference and training would "
            "resize to different pixels"
        )
    except Exception:
        failures += 1
        traceback.print_exc()
        print("\nPREFLIGHT FAILED (config) -- cannot continue")
        return 1

    # -- 3. samples -------------------------------------------------------
    # Set by block 3, read by 3b; defined before the try so a failure reads as a failure.
    expected_total = None
    print("\n== dataset ==")
    try:
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset

        ds = get_vla_dataset(data_cfg=vla)
        n = len(ds)
        print(f"  len(dataset) = {n}")
        assert n > 0, "dataset is empty"
        # Not an exact count -- episode lengths may change -- but the raw frame total means no trimming.
        assert n < 36962, (
            f"len={n} equals the raw frame count, so no chunk was trimmed: the "
            f"episode-boundary cut is not being applied"
        )

        # The mixture's __getitem__(i) draws a random step, so these are independent draws.
        idxs = sorted({0, n - 1, n // 2, n // 4, 3 * n // 4})
        idxs = idxs[: max(1, args.samples)]
        print(f"  {len(idxs)} draws from the mixture (random steps, see note)")
        for i in idxs:
            s = ds[i]
            a = np.asarray(s["action"])
            imgs = s["image"]
            lang = s["lang"]
            assert a.shape == (horizon, 3), (
                f"index {i}: chunk shape {a.shape}, expected ({horizon}, 3)"
            )
            assert np.isfinite(a).all(), f"index {i}: non-finite action"
            assert len(imgs) == 1, f"index {i}: {len(imgs)} images, expected 1"
            assert imgs[0].size == tuple(vla.obs_image_size), (
                f"index {i}: image {imgs[0].size}, expected {tuple(vla.obs_image_size)}"
            )
            assert isinstance(lang, str) and lang.strip(), f"index {i}: no lang"
            print(
                f"    [{i:6d}] action{a.shape} {a.dtype} "
                f"img{imgs[0].size} lang={lang[:34]!r}"
            )

        # Re-derived from index.jsonl: a chunk straddling an episode still has the right shape.
        import json
        from collections import Counter

        root = Path(vla.data_root_dir) / "sft_data"
        per_ep = Counter()
        with open(root / "index.jsonl") as fh:
            for line in fh:
                if line.strip():
                    per_ep[int(json.loads(line)["episode"])] += 1
        expected_total = sum(L - horizon + 1 for L in per_ep.values() if L >= horizon)
        print(
            f"  episodes={len(per_ep)} frames={sum(per_ep.values())} "
            f"total_steps_over_all_episodes={expected_total}"
        )
        # Upper bound only: expected_total spans all episodes, len(ds) is one split of them.
        assert n <= expected_total, (
            f"len(ds)={n} exceeds the {expected_total} steps the raw index "
            f"supports: the episode-boundary arithmetic is over-counting"
        )
        print(
            f"  boundary: len(ds)={n} <= {expected_total} = "
            f"sum(L - {horizon} + 1) over {len(per_ep)} episodes"
        )

        # The edges, where the index is faithful: a chunk running off the end has the right shape.
        inner = ds.datasets[0]
        for i in (0, len(inner) - 1):
            s_i = inner[i]
            assert np.asarray(s_i["action"]).shape == (horizon, 3), (
                f"inner[{i}] chunk {np.asarray(s_i['action']).shape}"
            )
            assert np.isfinite(np.asarray(s_i["action"])).all(), (
                f"inner[{i}] has a non-finite action"
            )
        last_ep, last_st = inner.all_steps[-1]
        assert last_st + horizon <= len(inner._episode_images[last_ep]), (
            f"the last training step (episode {last_ep} step {last_st}) needs "
            f"{horizon} actions but that episode has "
            f"{len(inner._episode_images[last_ep])}: the final chunk runs past "
            f"the end of its episode"
        )
        print(
            f"  edges: inner[0] and inner[{len(inner) - 1}] are full chunks; "
            f"last step is episode {last_ep} step {last_st} of "
            f"{len(inner._episode_images[last_ep])}"
        )
    except Exception:
        failures += 1
        traceback.print_exc()

    # -- 3b. the held-out split, and the normalisation --------------------
    # A wrong inverse reads as "SFT does not drive", not as "the units are wrong".
    print("\n== split + normalisation ==")
    try:
        # Loaded from a path: the copy that actually runs is the deployed registry copy.
        import importlib.util

        dc_path = Path(args.data_config)
        spec = importlib.util.spec_from_file_location("carla_data_config", dc_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        normalize_actions = mod.normalize_actions
        unnormalize_actions = mod.unnormalize_actions
        print(f"  loaded adapter from {dc_path}")

        holdout = int(vla.get("holdout_episodes", 0))
        eval_cfg = OmegaConf.create(OmegaConf.to_container(vla, resolve=True))
        eval_cfg.split = "eval"
        ds_eval = get_vla_dataset(data_cfg=eval_cfg)

        n_train, n_eval = len(ds), len(ds_eval)
        print(
            f"  holdout_episodes={holdout}  train_len={n_train}  "
            f"eval_len={n_eval}  sum={n_train + n_eval}"
        )
        assert expected_total is not None, (
            "the previous block failed, so there is no independently derived "
            "total to check the split against"
        )
        assert n_train + n_eval == expected_total, (
            f"train {n_train} + eval {n_eval} = {n_train + n_eval} but the raw "
            f"index supports {expected_total}: the split is not a partition, "
            f"so steps are double-counted or dropped"
        )
        assert holdout > 0, "holdout_episodes is 0: there is no held-out set"

        # get_vla_dataset wraps ours in a mixture, so the episode lists are one level down.
        inner, inner_eval = ds.datasets[0], ds_eval.datasets[0]
        tr, ev = set(inner.split_episodes), set(inner_eval.split_episodes)
        assert not (tr & ev), (
            f"episodes in both splits: {sorted(tr & ev)} -- a step-level leak "
            f"would look exactly like this"
        )
        # Episodes shorter than the horizon hold no valid chunk: absent from both splits.
        print(
            f"  disjoint: {len(tr)} train episodes, {len(ev)} held-out ({sorted(ev)})"
        )

        # The public accessor: these are the numbers a rollout undoes normalisation with.
        stats_tr = inner.get_dataset_statistics()[inner.dataset_name]["action"]
        stats_ev = inner_eval.get_dataset_statistics()[inner_eval.dataset_name][
            "action"
        ]
        for k in ("min", "max"):
            assert np.array_equal(stats_tr[k], stats_ev[k]), (
                f"normalisation {k} differs between splits; the eval set is "
                f"not in the units training used"
            )

        # Round trip: the loose tolerance catches a wrong formula, not float32 rounding.
        raw = inner._episode_actions[inner.all_steps[0][0]][:horizon]
        back = unnormalize_actions(normalize_actions(raw, stats_tr), stats_tr)
        err = float(np.abs(raw - back).max())
        print(f"  round-trip max abs error = {err:.2e} (float32 rounding scale)")
        assert err < 1e-5, (
            f"normalise/unnormalise do not round-trip; max error {err:.2e} is "
            f"far above float32 rounding, so one of the two formulas is wrong"
        )

        # Paired through the inner dataset: the mixture's __getitem__(i) draws a random step.
        mid = n_train // 2
        ep_mid, st_mid = inner.all_steps[mid]
        a_norm = np.asarray(inner[mid]["action"])
        raw_mid = inner._episode_actions[ep_mid][st_mid : st_mid + horizon]
        print(
            f"  step [{mid}] normalised {a_norm.min():.3f}..{a_norm.max():.3f}"
            f"  <- raw {raw_mid.min():.3f}..{raw_mid.max():.3f}"
        )
        assert a_norm.min() >= -1.0001 and a_norm.max() <= 1.0001, (
            f"normalised action outside [-1, 1]: {a_norm.min()}..{a_norm.max()}"
        )

        # The mixture must also yield a well-formed sample; only its shape is checkable.
        assert np.asarray(ds[mid]["action"]).shape == (horizon, 3), (
            "the mixture returned a malformed sample"
        )

        # A split returning the right count but the wrong samples would otherwise surface later.
        ev_mid = n_eval // 2
        s_ev = ds_eval[ev_mid]
        a_ev = np.asarray(s_ev["action"])
        assert a_ev.shape == (horizon, 3), (
            f"eval sample {ev_mid}: chunk {a_ev.shape}, expected {(horizon, 3)}"
        )
        assert s_ev["image"][0].size == tuple(vla.obs_image_size), (
            f"eval sample: image {s_ev['image'][0].size}"
        )
        ep_ev = inner_eval.all_steps[ev_mid][0]
        assert ep_ev in ev, (
            f"eval sample {ev_mid} came from episode {ep_ev}, which is not in "
            f"the held-out set {sorted(ev)}"
        )
        print(
            f"  eval sample [{ev_mid}] from episode {ep_ev} "
            f"action{a_ev.shape} img{s_ev['image'][0].size}"
        )
    except Exception:
        failures += 1
        traceback.print_exc()

    # -- 4. save_dataset_statistics ---------------------------------------
    # Needs no GPU, but it is the first thing build_dataloader does, after the model is built.
    print("\n== save_dataset_statistics ==")
    try:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dataset_statistics.json"
            ds.save_dataset_statistics(path)
            blob = json.loads(path.read_text())
        keys = list(blob)
        print(f"  wrote {path.name}: {len(blob)} tag(s) {keys}")
        assert keys, "statistics file is empty"
        # An empty "action" section means the modality subkey did not match the mixture's.
        tag_blob = blob[keys[0]]
        assert tag_blob.get("action"), (
            f"no action statistics under tag {keys[0]}: {list(tag_blob)}"
        )
        # These numbers are what inference de-normalizes predictions with, so a
        # wrong-length or non-finite vector here is a real defect, not a nit.
        for stat in ("mean", "std"):
            v = tag_blob["action"].get(stat)
            assert v is not None, f"action.{stat} missing from the saved stats"
            assert len(v) == 3, f"action.{stat} has {len(v)} dims, expected 3"
            assert all(np.isfinite(v)), f"action.{stat} is not finite: {v}"
        print(
            f"  action.mean = {[round(float(x), 5) for x in tag_blob['action']['mean']]}"
        )
        print(
            f"  action.std  = {[round(float(x), 5) for x in tag_blob['action']['std']]}"
        )
        print(
            f"  counts      = "
            f"{ {k: v for k, v in tag_blob.items() if k not in ('action', 'state')} }"
        )
    except Exception:
        failures += 1
        traceback.print_exc()

    print()
    if failures:
        print(f"PREFLIGHT FAILED ({failures} block(s))")
    else:
        print("PREFLIGHT OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
