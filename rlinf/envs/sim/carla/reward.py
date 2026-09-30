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

"""Reward for the CARLA GRPO runs; Bench2Drive's SR as the terminal signal, not its DS."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Outcome(enum.Enum):
    """Why an episode ended. Only the terminal values carry reward."""

    RUNNING = "running"
    SUCCESS = "success"
    CRASH = "crash"
    TIMEOUT = "timeout"
    #: Ended without succeeding or crashing; scored like a timeout, not a crash.
    INCOMPLETE = "incomplete"


#: Infractions that map to Outcome.CRASH and its negative terminal reward.
CRASH_INFRACTIONS = frozenset(
    {
        "collisions_pedestrian",
        "collisions_vehicle",
        "collisions_layout",
        "red_light",
        "stop_infraction",
        "outside_route_lanes",
        "vehicle_blocked",
        "route_dev",
    }
)

#: Bench2Drive exempts this one from the success check; hence the named constant.
MIN_SPEED_INFRACTION = "min_speed_infractions"


@dataclass
class RewardConfig:
    """Knobs, all in one place so a run's reward is readable off its yaml."""

    #: "sparse" -> terminal only; "dense" -> terminal + dRC/100 - c per step.
    mode: str = "sparse"

    success_bonus: float = 1.0
    crash_penalty: float = -1.0
    #: Non-crash endings; 0.0, not a penalty: penalising termination yields "never move".
    incomplete_reward: float = 0.0

    #: Per-step time cost c, dense mode only; None derives it from the route length.
    time_cost: float | None = None

    #: Fallback denominator when time_cost is derived and the route length is unknown.
    assumed_route_steps: int = 1200

    def __post_init__(self) -> None:
        if self.mode not in ("sparse", "dense"):
            raise ValueError(f"mode must be 'sparse' or 'dense', got {self.mode!r}")
        if self.mode == "dense" and self.time_cost is not None:
            if self.time_cost < 0:
                raise ValueError("time_cost is a cost; pass it >= 0")

    def effective_time_cost(self, route_steps: int | None = None) -> float:
        """c, derived from the route length when not set explicitly."""
        if self.time_cost is not None:
            return self.time_cost
        steps = route_steps or self.assumed_route_steps
        if steps <= 0:
            return 0.0
        return 1.0 / steps


@dataclass
class EpisodeState:
    """What the reward needs to know about an episode in flight."""

    route_completion: float = 0.0
    infractions: set[str] = field(default_factory=set)

    def is_success(self) -> bool:
        """Bench2Drive's SR, faithfully: RC >= 100 with no infraction, min_speed exempted."""
        if self.route_completion < 100.0:
            return False
        return not any(i != MIN_SPEED_INFRACTION for i in self.infractions)

    def has_crashed(self) -> bool:
        return bool(self.infractions & CRASH_INFRACTIONS)


class CarlaReward:
    """Stateful per-episode reward; reset() at every episode end or the delta carries over."""

    def __init__(self, cfg: RewardConfig | None = None) -> None:
        self.cfg = cfg or RewardConfig()
        self._prev_rc = 0.0
        self._c = self.cfg.effective_time_cost()

    @property
    def time_cost(self) -> float:
        return self._c

    def set_route_steps(self, route_steps: int) -> None:
        """Give the reward the real route length before the episode; dense mode only."""
        self._c = self.cfg.effective_time_cost(route_steps)

    def reset(self) -> None:
        self._prev_rc = 0.0

    def step(
        self,
        route_completion: float,
        outcome: Outcome = Outcome.RUNNING,
    ) -> float:
        """One env tick; returns that tick's scalar reward for a 0-100 route_completion."""
        # A replan can nudge RC backwards, so clamp the increment and never lower _prev_rc.
        rc = max(0.0, min(100.0, float(route_completion)))

        reward = 0.0
        if self.cfg.mode == "dense":
            reward += max(0.0, rc - self._prev_rc) / 100.0
            reward -= self._c

        self._prev_rc = max(self._prev_rc, rc)

        if outcome is Outcome.SUCCESS:
            reward += self.cfg.success_bonus
        elif outcome is Outcome.CRASH:
            reward += self.cfg.crash_penalty
        elif outcome in (Outcome.TIMEOUT, Outcome.INCOMPLETE):
            reward += self.cfg.incomplete_reward

        return reward

    def classify(self, state: EpisodeState, timed_out: bool) -> Outcome:
        """Map an episode's final state onto an Outcome; a crash that finished is a crash."""
        if state.has_crashed():
            return Outcome.CRASH
        if state.is_success():
            return Outcome.SUCCESS
        if timed_out:
            return Outcome.TIMEOUT
        return Outcome.INCOMPLETE
