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

"""CARLA integration tests that do not require a simulator or GPU."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


class CarlaIntegrationTest(unittest.TestCase):
    def test_config_resolves_environment_paths(self):
        checker = load_module(
            "carla_config_test",
            ROOT / "examples/embodiment/carla/verify_env_config.py",
        )
        config = {"model": {"path": "${oc.env:CARLA_SFT_CHECKPOINT}"}}
        with patch.dict("os.environ", {"CARLA_SFT_CHECKPOINT": "/tmp/model.pt"}):
            self.assertEqual(checker._resolve("${model.path}", config), "/tmp/model.pt")

    def test_grpo_configs_keep_group_layout(self):
        config_dir = ROOT / "examples/embodiment/config"
        expected = {
            "carla_grpo_starvla.yaml": (16, 4, 4),
            "carla_grpo_starvla_async.yaml": (16, 4, 4),
            "carla_grpo_starvla_async_split.yaml": (12, 3, 3),
        }
        for filename, (num_envs, env_ranks, actor_ranks) in expected.items():
            with self.subTest(filename=filename):
                config = yaml.safe_load((config_dir / filename).read_text())
                self.assertEqual(config["algorithm"]["group_size"], 4)
                self.assertEqual(config["env"]["train"]["total_num_envs"], num_envs)
                self.assertEqual(num_envs // env_ranks, 4)
                self.assertEqual(num_envs // actor_ranks, 4)
                self.assertEqual(
                    config["rollout"]["model"]["model_path"],
                    "${oc.env:CARLA_SFT_CHECKPOINT}",
                )

    def test_route_progress(self):
        route = load_module("carla_route_test", ROOT / "rlinf/envs/sim/carla/route.py")
        path = route.Route(((0.0, 0.0), (10.0, 0.0)))
        self.assertEqual(path.progress(5.0, 0.0), 50.0)
        self.assertEqual(path.progress(10.0, 0.0), 100.0)

    def test_three_dim_action_unnormalizes_without_gripper(self):
        action_space = load_module(
            "carla_action_space_test",
            ROOT / "rlinf/models/embodiment/starvla/utils/action_space.py",
        )
        framework = types.ModuleType("starVLA.model.framework.base_framework")

        class BaseFramework:
            @staticmethod
            def unnormalize_actions(actions, stats):
                raise AssertionError(
                    "three-channel actions must not use the gripper helper"
                )

        framework.baseframework = BaseFramework
        modules = {
            "starVLA": types.ModuleType("starVLA"),
            "starVLA.model": types.ModuleType("starVLA.model"),
            "starVLA.model.framework": types.ModuleType("starVLA.model.framework"),
            "starVLA.model.framework.base_framework": framework,
        }
        stats = {
            "q01": np.array([-0.8, 0.0, 0.0], dtype=np.float32),
            "q99": np.array([0.8, 0.75, 1.0], dtype=np.float32),
            "mask": np.array([True, True, True]),
        }
        actions = np.array([[[0.0, 1.0, -1.0]]], dtype=np.float32)
        with patch.dict(sys.modules, modules):
            result = action_space.unnormalize_actions_for_env(actions, stats)
        np.testing.assert_allclose(result, [[[0.0, 0.75, 0.0]]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
