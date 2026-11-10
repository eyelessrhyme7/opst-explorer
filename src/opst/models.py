from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Waypoint:
    """A candidate station/task; depot metadata lives on the Instance."""

    waypoint_id: str
    lat: float
    lon: float
    service_time_s: int
    reward: int
    h3_cell: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Instance:
    """Canonical OP-ST instance.

    Matrix row/column 0 is the depot. Matrix indices 1..n correspond to
    ``waypoints`` in order.
    """

    instance_id: str
    depot_id: str
    depot_lat: float
    depot_lon: float
    waypoints: tuple[Waypoint, ...]
    travel_time_s: NDArray[np.int64]
    budget_s: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.waypoints)
        matrix = np.asarray(self.travel_time_s, dtype=np.int64)
        object.__setattr__(self, "travel_time_s", matrix)
        if matrix.shape != (n + 1, n + 1):
            raise ValueError(
                f"travel_time_s shape {matrix.shape} does not match depot + {n} waypoints"
            )
        if self.budget_s < 0:
            raise ValueError("budget_s must be nonnegative")
        for waypoint in self.waypoints:
            if waypoint.service_time_s < 0:
                raise ValueError(f"{waypoint.waypoint_id} has negative service time")
            if waypoint.reward < 0:
                raise ValueError(f"{waypoint.waypoint_id} has negative reward")


@dataclass(frozen=True)
class Solution:
    """Canonical solver output. Values are independently checked by the validator."""

    instance_id: str
    algorithm: str
    route_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    reward: int
    travel_time_s: int
    service_time_s: int
    total_time_s: int
    feasible: bool
    runtime_ms: float
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
