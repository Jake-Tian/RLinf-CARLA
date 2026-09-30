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

"""StarVLA data registry entry for the CARLA SFT dataset.

StarVLA's action path reaches past `dataset[i]` into get_step_data/transforms/_pack_sample.
Registered through the `make_dataset` hook; no file in StarVLA-SFT is modified.
`action` is 3-dim control (steer, throttle, brake), not UniDriveVLA's 2-dim trajectory.
Chunks never straddle an episode: the last horizon-1 steps of each are dropped.
One pseudo-trajectory of len(all_steps): sample_step picks a flat base_index into it.
The head predicts *normalised* actions; rollouts must call `unnormalize_actions` first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

#: Control channels, in the order `collect_sft_data.py` writes them.
ACTION_DIM = 3
#: Must match framework.action_model.action_horizon in the YAML; the chunk is sliced here.
ACTION_HORIZON = 6
#: PIL takes (width, height); matches obs_image_size in the YAML, which QwenOFT resizes to.
IMAGE_SIZE = (640, 360)

#: Modality key names; the values are opaque labels as long as get_step_data returns them.
VIDEO_KEY = "video.ego_view"
LANGUAGE_KEY = "language.instruction"
ACTION_KEY = "action.control"

#: The `action.` prefix stripped, as the statistics dicts are keyed; a wrong one is silent.
ACTION_SUBKEY = ACTION_KEY.split(".", 1)[1]  # "control"

#: Metadata only: 20 fps is what the collector ran at.
FPS = 20.0


def normalize_actions(actions, stats):
    """Scale to [-1, 1] with `min_max`, the mode the RoboCasa365 config this run is modelled on."""
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    span = hi - lo
    degenerate = span == 0
    safe = np.where(degenerate, 1.0, span)
    out = 2.0 * (actions - lo) / safe - 1.0
    # Where min == max StarVLA writes 0 rather than the raw value.
    return np.where(degenerate, 0.0, out)


def unnormalize_actions(actions, stats):
    """The inverse; any rollout code must apply this to the model's raw output."""
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    return (actions + 1.0) / 2.0 * (hi - lo) + lo


def _normalise_tag(embodiment_tag):
    """`EmbodimentTag.NEW_EMBODIMENT` -> its string value.

    save_dataset_statistics uses the tag as a JSON object key; an enum dies there.
    """
    return getattr(embodiment_tag, "value", embodiment_tag)


class _NoTransforms:
    """The pipeline the mixture expects to reach into; deliberately empty, not a stub.

    StarVLA *iterates* `transforms` to discover per-key normalization modes; empty means none.
    """

    def __init__(self):
        self.transforms = []

    def __call__(self, raw_data):
        return raw_data

    def set_metadata(self, metadata):
        self._metadata = metadata


class CarlaSFTDataset:
    """Reads one `sft_data` directory as a StarVLA-compatible dataset.

    Not a subclass: LeRobotSingleDataset.__init__ reads metadata files this dataset lacks.
    """

    def __init__(
        self,
        dataset_path,
        data_cfg=None,
        embodiment_tag=None,
        action_horizon=ACTION_HORIZON,
        image_size=IMAGE_SIZE,
        split="train",
        holdout_episodes=0,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.data_cfg = data_cfg
        # A string, not the enum: see _normalise_tag.
        self.tag = _normalise_tag(embodiment_tag)
        self.action_horizon = int(action_horizon)
        self.image_size = tuple(image_size)
        if split not in ("train", "eval"):
            raise ValueError(
                f"split={split!r}; expected 'train' or 'eval'. Training must "
                f"not silently fall back to the full dataset -- that would "
                f"train on the held-out routes and make the eval meaningless"
            )
        self.split = split
        self.holdout_episodes = int(holdout_episodes)

        index_path = self.dataset_path / "index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"no index.jsonl in {self.dataset_path}; the CARLA collector "
                f"writes one per run -- check the dataset path"
            )
        self.transforms = _NoTransforms()
        self._load(index_path)
        # Built after _load because it is computed from the loaded actions.
        self._metadata = self._build_metadata()

    # -- loading -----------------------------------------------------------

    def _load(self, index_path):
        """Group the flat index by episode and build the flat step table.

        Actions are stored per episode as one `(L, ACTION_DIM)` array; a chunk is a slice.
        """
        episodes = {}
        with open(index_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep = int(rec["episode"])
                episodes.setdefault(ep, []).append(rec)

        self.episode_ids = sorted(episodes)
        self._episode_actions = {}
        self._episode_images = {}
        self._episode_tasks = {}

        for ep in self.episode_ids:
            recs = sorted(episodes[ep], key=lambda r: int(r["step"]))
            actions = np.asarray([r["action"] for r in recs], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"episode {ep}: expected actions shaped (L, {ACTION_DIM}), "
                    f"got {actions.shape}"
                )
            self._episode_actions[ep] = actions
            self._episode_images[ep] = [r["image"] for r in recs]
            # The instruction is per-episode, taken from the first record; a mid-episode change is a bug.
            self._episode_tasks[ep] = recs[0]["task"]

        # -- the episode split --------------------------------------------
        lengths = {ep: len(self._episode_images[ep]) for ep in self.episode_ids}
        self._holdout_episodes = self._select_holdout(lengths)
        self._train_episodes = [
            ep for ep in self.episode_ids if ep not in set(self._holdout_episodes)
        ]
        self.split_episodes = (
            self._holdout_episodes if self.split == "eval" else self._train_episodes
        )

        # Normalisation constants come from the TRAIN episodes in *both* splits.
        self._norm_stats = self._compute_action_stats(self._train_episodes)

        self.all_steps = []
        for ep in self.split_episodes:
            # A chunk never straddles an episode: the last horizon-1 steps are dropped, not padded.
            for s in range(lengths[ep] - self.action_horizon + 1):
                self.all_steps.append((ep, s))

        if not self.all_steps:
            raise ValueError(
                f"{self.dataset_path}: split={self.split!r} selected "
                f"{len(self.split_episodes)} episodes and none of them is "
                f"longer than action_horizon={self.action_horizon}, so there "
                f"is nothing to train or score on"
            )

        # One pseudo-trajectory: base_index is a flat index into all_steps.
        self.trajectory_ids = [0]
        self.trajectory_lengths = np.asarray([len(self.all_steps)], dtype=np.int64)

        # The mixture gates sampling on physical video files; this dataset is JPEG frames.
        self.lerobot_info_meta = {"total_videos": 0}
        self.modality_keys = {
            "video": [VIDEO_KEY],
            "language": [LANGUAGE_KEY],
            "action": [ACTION_KEY],
        }
        # Only consulted when drop_incomplete_action_chunks is set; chunks are complete anyway.
        self.delta_indices = {"action": np.arange(self.action_horizon)}

        n_frames = int(sum(len(self._episode_images[ep]) for ep in self.split_episodes))
        print(
            f"[carla_sft/{self.split}] {self.dataset_name}: "
            f"{len(self.split_episodes)} of {len(self.episode_ids)} episodes, "
            f"{n_frames} frames, {len(self.all_steps)} steps "
            f"(horizon={self.action_horizon}, image={self.image_size}, "
            f"holdout={len(self._holdout_episodes)})"
        )
        if self._holdout_episodes:
            print(
                f"[carla_sft/{self.split}] held-out episodes "
                f"{self._holdout_episodes} "
                f"({'excluded from' if self.split == 'train' else 'only in'} "
                f"this split)"
            )

    def _select_holdout(self, lengths):
        """Which episodes are held out, deterministically.

        Whole episodes, not steps: at 20 fps neighbouring steps of one route are near-identical.
        Evenly spread across the index, not the last N, which could be one weather or town.
        Only *full-length* episodes are eligible: a route is never one already truncated.
        """
        n = self.holdout_episodes
        if n <= 0:
            return []
        longest = max(lengths.values())
        full = [ep for ep in self.episode_ids if lengths[ep] == longest]
        if n > len(full):
            raise ValueError(
                f"holdout_episodes={n} but only {len(full)} episodes are "
                f"full-length ({longest} frames); holding out more would have "
                f"to start including truncated routes"
            )
        idx = np.linspace(0, len(full) - 1, n).round().astype(int)
        return sorted({full[i] for i in idx})

    # -- the surface LeRobotMixtureDataset calls ---------------------------

    def __len__(self):
        return len(self.all_steps)

    @property
    def metadata(self):
        """What `LeRobotMixtureDataset.update_metadata` reads to merge stats.

        A real `DatasetMetadata`, not a bare namespace: the mixture re-validates the merge.
        """
        return self._metadata

    def set_transforms_metadata(self, metadata):
        """Called with the *merged* metadata; forwarded to the empty pipeline and ignored."""
        self.transforms.set_metadata(metadata)

    def _build_metadata(self):
        # Imported here, not at module scope: this file imports without StarVLA on sys.path.
        from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
        from starVLA.dataloader.gr00t_lerobot.schema import (
            DatasetMetadata,
            DatasetModalities,
            DatasetStatisticalValues,
            DatasetStatistics,
            StateActionMetadata,
            VideoMetadata,
        )

        s = self.get_dataset_statistics()[self.dataset_name]["action"]
        # `state` is empty on purpose: an empty dict makes compute_overall_statistics skip it.
        return DatasetMetadata(
            statistics=DatasetStatistics(
                state={},
                action={
                    ACTION_SUBKEY: DatasetStatisticalValues(
                        **{
                            k: np.asarray(s[k], dtype=np.float32)
                            for k in ("max", "min", "mean", "std", "q01", "q99")
                        }
                    )
                },
            ),
            modalities=DatasetModalities(
                # [width, height], the order StarVLA writes for a LeRobot dataset.
                video={
                    VIDEO_KEY.split(".", 1)[1]: VideoMetadata(
                        resolution=self.image_size, channels=3, fps=FPS
                    )
                },
                state={},
                # Steer, throttle and brake are absolute and continuous; nothing is masked out.
                action={
                    ACTION_SUBKEY: StateActionMetadata(
                        absolute=True,
                        rotation_type=None,
                        shape=(ACTION_DIM,),
                        continuous=True,
                    )
                },
            ),
            # The enum, not the string: this field is typed EmbodimentTag and compared across datasets.
            embodiment_tag=EmbodimentTag(self.tag),
        )

    def get_step_data(self, trajectory_id, base_index):
        """Raw record for a flat step index.

        `trajectory_id` is asserted to be 0: a non-zero id means the sampler and this table drifted.
        """
        if int(trajectory_id) != 0:
            raise ValueError(
                f"trajectory_id={trajectory_id!r} but this dataset exposes a "
                f"single pseudo-trajectory; trajectory_lengths="
                f"{self.trajectory_lengths.tolist()}"
            )
        i = int(base_index)
        if not 0 <= i < len(self.all_steps):
            raise IndexError(
                f"base_index {i} outside [0, {len(self.all_steps)}); the "
                f"sampler and this dataset disagree about the dataset length"
            )
        ep, step = self.all_steps[i]
        return {
            VIDEO_KEY: self._episode_images[ep][step],
            LANGUAGE_KEY: self._episode_tasks[ep],
            ACTION_KEY: (ep, step),
            "_episode": ep,
        }

    def _pack_sample(self, data):
        """Decode the JPEG and slice the action chunk, returning the dict QwenOFT.forward reads.

        `lang` is the bare instruction: QwenOFT appends its own "Please predict..." suffix.
        """
        ep, step = data[ACTION_KEY]
        actions = self._episode_actions[ep][step : step + self.action_horizon]
        if actions.shape[0] != self.action_horizon:
            raise ValueError(
                f"episode {ep} step {step}: chunk has "
                f"{actions.shape[0]} steps, expected {self.action_horizon}; "
                f"all_steps should not contain a step this close to the end"
            )

        image_path = self.dataset_path / data[VIDEO_KEY]
        with Image.open(image_path) as img:
            image = img.convert("RGB").resize(self.image_size, Image.BILINEAR)

        return {
            # float32, not the loader's float16 default: this is an L1 regression target.
            # Normalised, so anything consuming this label must call `unnormalize_actions`.
            "action": np.ascontiguousarray(
                normalize_actions(actions, self._norm_stats), dtype=np.float32
            ),
            "image": [image],
            "lang": data[LANGUAGE_KEY],
            "robot_tag": self.tag,
        }

    def __getitem__(self, index):
        """Standalone access for `python -m` debugging; the training path does not use it."""
        return self._pack_sample(self.transforms(self.get_step_data(0, index)))

    # -- statistics --------------------------------------------------------

    def _compute_action_stats(self, episodes):
        """Per-dimension action statistics over the given episodes.

        Counted per *timestep*, not per chunk: the horizon does not reweight anything.
        The single source of the units: `_pack_sample` normalises with these numbers.
        """
        all_actions = np.concatenate(
            [self._episode_actions[ep] for ep in episodes], axis=0
        )
        return {
            "mean": all_actions.mean(axis=0),
            "std": all_actions.std(axis=0),
            "min": all_actions.min(axis=0),
            "max": all_actions.max(axis=0),
            "q01": np.quantile(all_actions, 0.01, axis=0),
            "q99": np.quantile(all_actions, 0.99, axis=0),
        }

    def get_dataset_statistics(self):
        """In the shape `save_dataset_statistics` expects.

        Always the train split's statistics: the file is identical whichever split asked.
        `num_transitions` counts training steps, not frames, because it must match `len(dataset)`.
        """
        train_steps = sum(
            len(self._episode_images[ep]) - self.action_horizon + 1
            for ep in self._train_episodes
            if len(self._episode_images[ep]) >= self.action_horizon
        )
        return {
            self.dataset_name: {
                "action": self._norm_stats,
                "num_transitions": int(train_steps),
                "num_trajectories": len(self._train_episodes),
            }
        }


class CarlaDataConfig:
    """DataConfig for `robot_type="carla_sft"`.

    `modality_config()` and `transform()` must exist: they are called before the hook.
    They return empty values rather than plausible ones that suggest the LeRobot path.
    """

    #: No `embodiment_tag` classvar on purpose: absent, NEW_EMBODIMENT is substituted.
    def modality_config(self):
        return {}

    def transform(self):
        return None

    def make_dataset(
        self,
        dataset_path,
        modality_configs=None,
        transforms=None,
        embodiment_tag=None,
        video_backend=None,
        delete_pause_frame=False,
        data_cfg=None,
        dataset_name=None,
    ):
        """The hook `make_LeRobotSingleDataset` looks for; chunk and image size come from config."""
        horizon = ACTION_HORIZON
        image_size = IMAGE_SIZE
        split = "train"
        holdout = 0
        if data_cfg is not None:
            horizon = int(data_cfg.get("action_horizon", horizon))
            size = data_cfg.get("image_size", None)
            if size is not None:
                image_size = tuple(size) if not isinstance(size, int) else (size, size)
            # `split` defaults to train and `holdout_episodes` to 0: silence means the full dataset.
            split = str(data_cfg.get("split", split))
            holdout = int(data_cfg.get("holdout_episodes", holdout))
        return CarlaSFTDataset(
            dataset_path=dataset_path,
            data_cfg=data_cfg,
            embodiment_tag=embodiment_tag,
            action_horizon=horizon,
            image_size=image_size,
            split=split,
            holdout_episodes=holdout,
        )


ROBOT_TYPE_CONFIG_MAP = {
    "carla_sft": CarlaDataConfig(),
}

#: mixture name -> [(dataset dir name, weight, robot_type)]; data_root_dir is its parent.
DATASET_NAMED_MIXTURES = {
    "carla_sft": [("sft_data", 1.0, "carla_sft")],
}
