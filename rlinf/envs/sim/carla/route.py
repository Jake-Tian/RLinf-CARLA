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

"""Route representation and progress-along-route; pure geometry, no CARLA dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    """An ordered polyline in world coordinates, metres; a tuple so it can be shared."""

    waypoints: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("a route needs at least two waypoints")

    # -- construction ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> Route:
        """Load whitespace/comma separated ``x y`` pairs, one per line; ``#`` starts a comment."""
        pts: list[tuple[float, float]] = []
        with open(path) as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    raise ValueError(f"{path}:{lineno}: expected 'x y', got {raw!r}")
                pts.append((float(parts[0]), float(parts[1])))
        return cls(tuple(pts))

    # -- geometry ----------------------------------------------------------

    @property
    def segment_lengths(self) -> list[float]:
        return [
            math.dist(self.waypoints[i], self.waypoints[i + 1])
            for i in range(len(self.waypoints) - 1)
        ]

    @property
    def length_m(self) -> float:
        return sum(self.segment_lengths)

    def project(self, x: float, y: float) -> tuple[float, float, int]:
        """Nearest point on the polyline as (distance_along_m, lateral_offset_m, segment_index)."""
        best = (0.0, float("inf"), 0)
        travelled = 0.0
        for i, (ax, ay) in enumerate(self.waypoints[:-1]):
            bx, by = self.waypoints[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0.0:
                continue
            t = ((x - ax) * dx + (y - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))  # clamp: only the segment, not the line
            px, py = ax + t * dx, ay + t * dy
            dist = math.dist((x, y), (px, py))
            if dist < best[1] - 1e-9:
                best = (travelled + t * math.sqrt(seg_len_sq), dist, i)
            travelled += math.sqrt(seg_len_sq)
        return best

    def progress(self, x: float, y: float) -> float:
        """Percentage of the route completed, 0-100; exactly 100.0 at the final waypoint."""
        along, _, _ = self.project(x, y)
        total = self.length_m
        if total <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * along / total))

    def lateral_offset(self, x: float, y: float) -> float:
        """Perpendicular distance from the route, in metres; what the off-route check keys on."""
        return self.project(x, y)[1]

    def is_off_route(self, x: float, y: float, max_offset_m: float) -> bool:
        return self.lateral_offset(x, y) > max_offset_m


def straight_route(
    origin: tuple[float, float],
    heading_rad: float,
    length_m: float,
    step_m: float = 2.0,
) -> Route:
    """A straight route for smoke tests, not a general route builder."""
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    n = max(2, int(math.ceil(length_m / step_m)) + 1)
    ox, oy = origin
    dx, dy = math.cos(heading_rad), math.sin(heading_rad)
    return Route(
        tuple(
            (ox + dx * (i * length_m / (n - 1)), oy + dy * (i * length_m / (n - 1)))
            for i in range(n)
        )
    )
