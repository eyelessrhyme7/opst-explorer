from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opst.models import Instance, Waypoint
from opst.route_ops import route_selected_by_cheapest_insertion
from opst.validate import route_totals

BUDGET_FRACTIONS = {"tight": 0.35, "medium": 0.50, "loose": 0.65}


@dataclass(frozen=True)
class MatrixStation:
    matrix_index: int
    role: str
    station_id: str
    name: str
    lat: float
    lon: float
    capacity: int
    source_snapshot: str


def load_matrix_stations(path: Path) -> tuple[MatrixStation, tuple[MatrixStation, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stations = tuple(
        MatrixStation(
            matrix_index=int(row["matrix_index"]),
            role=row["role"],
            station_id=row["station_id"],
            name=row["name"],
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            capacity=int(row.get("capacity", 0) or 0),
            source_snapshot=row["source_snapshot"],
        )
        for row in rows
    )
    depots = tuple(row for row in stations if row.role == "depot")
    if len(depots) != 1:
        raise ValueError("station manifest must contain exactly one depot row")
    candidates = tuple(row for row in stations if row.role == "candidate")
    return depots[0], candidates


def _distance2(a: MatrixStation, b: MatrixStation) -> float:
    return (a.lat - b.lat) ** 2 + (a.lon - b.lon) ** 2


def _farthest_fill(
    selected: list[MatrixStation], pool: list[MatrixStation], target_n: int
) -> list[MatrixStation]:
    while len(selected) < target_n and len(selected) < len(pool):
        remaining = [station for station in pool if station not in selected]
        next_station = max(
            remaining,
            key=lambda station: (
                min(_distance2(station, chosen) for chosen in selected),
                station.station_id,
            ),
        )
        selected.append(next_station)
    return selected


def select_spatial_subset(
    candidates: tuple[MatrixStation, ...],
    *,
    n: int,
    regime: str,
    seed: int,
) -> tuple[MatrixStation, ...]:
    """Select matrix stations under a transparent synthetic spatial regime."""

    if n > len(candidates):
        raise ValueError(f"requested n={n}, but only {len(candidates)} candidates exist")
    rng = np.random.default_rng(seed)
    pool = list(candidates)
    anchor = pool[int(rng.integers(0, len(pool)))]
    compact_order = sorted(pool, key=lambda station: (_distance2(station, anchor), station.station_id))

    if regime == "compact":
        selected = compact_order[:n]
    elif regime == "dispersed":
        selected = _farthest_fill([anchor], pool, n)
    elif regime == "mixed":
        compact_n = max(1, math.ceil(0.7 * n))
        selected = compact_order[:compact_n]
        selected = _farthest_fill(selected, pool, n)
    elif regime == "outlier":
        compact_n = max(1, n - 1)
        selected = compact_order[:compact_n]
        selected = _farthest_fill(selected, pool, n)
    else:
        raise ValueError(f"unknown spatial regime: {regime}")
    return tuple(sorted(selected, key=lambda station: station.matrix_index))


def _h3_cell(lat: float, lon: float, resolution: int) -> str | None:
    try:
        import h3

        return str(h3.latlng_to_cell(lat, lon, resolution))
    except Exception:
        return None


def _task_values(
    rng: np.random.Generator,
    *,
    service_scale: float,
    outlier: bool,
    reward_model: str = "coarse",
) -> tuple[int, int, int, int]:
    task_count = int(max(1, rng.poisson(2.0)))
    urgency = int(rng.integers(1, 4))
    service_time_s = int(math.ceil((90 + 120 * task_count) * service_scale))
    if reward_model == "coarse":
        reward = int(task_count * urgency)
    elif reward_model == "fine":
        reward = int(10 * task_count + urgency)
    else:
        raise ValueError(f"unknown reward model: {reward_model}")
    if outlier:
        reward *= 4
    return task_count, urgency, service_time_s, reward


def generate_instance_from_matrix(
    *,
    matrix_id: str,
    depot: MatrixStation,
    candidates: tuple[MatrixStation, ...],
    travel_time_s: NDArray[np.int64],
    n: int,
    regime: str,
    service_scale: float,
    budget_tightness: str,
    seed: int,
    h3_resolution: int = 8,
    reward_model: str = "coarse",
) -> Instance:
    """Generate one seeded OP-ST instance from a cached road matrix."""

    selected = select_spatial_subset(candidates, n=n, regime=regime, seed=seed)
    selected_global_indices = [station.matrix_index for station in selected]
    matrix_indices = [0, *selected_global_indices]
    submatrix = travel_time_s[np.ix_(matrix_indices, matrix_indices)].astype(np.int64)

    rng = np.random.default_rng(seed)
    outlier_global = None
    if regime == "outlier" and len(selected) > 1:
        outlier_global = max(
            selected,
            key=lambda station: min(_distance2(station, other) for other in selected if other != station),
        ).matrix_index
    waypoints = []
    for station in selected:
        task_count, urgency, service_time_s, reward = _task_values(
            rng,
            service_scale=service_scale,
            outlier=station.matrix_index == outlier_global,
            reward_model=reward_model,
        )
        waypoints.append(
            Waypoint(
                waypoint_id=f"station_{station.station_id}",
                lat=station.lat,
                lon=station.lon,
                service_time_s=service_time_s,
                reward=reward,
                h3_cell=_h3_cell(station.lat, station.lon, h3_resolution),
                metadata={
                    "station_id": station.station_id,
                    "name": station.name,
                    "capacity": station.capacity,
                    "task_count": task_count,
                    "urgency": urgency,
                    "generator_regime": regime,
                    "seed": seed,
                    "source_matrix_index": station.matrix_index,
                },
            )
        )

    provisional = Instance(
        instance_id="provisional",
        depot_id=f"station_{depot.station_id}",
        depot_lat=depot.lat,
        depot_lon=depot.lon,
        waypoints=tuple(waypoints),
        travel_time_s=submatrix,
        budget_s=10**9,
        metadata={},
    )
    full_route = route_selected_by_cheapest_insertion(
        provisional, tuple(range(1, len(waypoints) + 1))
    )
    all_route_total_s = route_totals(provisional, full_route)[3]
    fraction = BUDGET_FRACTIONS[budget_tightness]
    individually_feasible_floor = min(
        int(2 * submatrix[0, idx] + waypoints[idx - 1].service_time_s)
        for idx in range(1, len(waypoints) + 1)
    )
    budget_s = max(int(math.ceil(fraction * all_route_total_s)), individually_feasible_floor)

    instance_id = (
        f"{matrix_id}_{regime}_n{n}_scale{service_scale:g}_{budget_tightness}_seed{seed}"
    )
    return Instance(
        instance_id=instance_id,
        depot_id=f"station_{depot.station_id}",
        depot_lat=depot.lat,
        depot_lon=depot.lon,
        waypoints=tuple(waypoints),
        travel_time_s=submatrix,
        budget_s=budget_s,
        metadata={
            "matrix_id": matrix_id,
            "source_matrix_indices": matrix_indices,
            "spatial_regime": regime,
            "service_scale": service_scale,
            "budget_tightness": budget_tightness,
            "budget_fraction_of_all_route": fraction,
            "all_route_total_s": all_route_total_s,
            "seed": seed,
            "h3_resolution": h3_resolution,
            "task_states": "synthetic",
            "reward_model": reward_model,
        },
    )


def instance_to_dict(instance: Instance) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "depot_id": instance.depot_id,
        "depot_lat": instance.depot_lat,
        "depot_lon": instance.depot_lon,
        "budget_s": instance.budget_s,
        "metadata": instance.metadata,
        "waypoints": [
            {
                "waypoint_id": waypoint.waypoint_id,
                "lat": waypoint.lat,
                "lon": waypoint.lon,
                "service_time_s": waypoint.service_time_s,
                "reward": waypoint.reward,
                "h3_cell": waypoint.h3_cell,
                "metadata": waypoint.metadata,
            }
            for waypoint in instance.waypoints
        ],
        "travel_time_s": instance.travel_time_s.astype(int).tolist(),
    }


def instance_from_dict(payload: dict[str, Any]) -> Instance:
    return Instance(
        instance_id=payload["instance_id"],
        depot_id=payload["depot_id"],
        depot_lat=float(payload["depot_lat"]),
        depot_lon=float(payload["depot_lon"]),
        waypoints=tuple(
            Waypoint(
                waypoint_id=row["waypoint_id"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                service_time_s=int(row["service_time_s"]),
                reward=int(row["reward"]),
                h3_cell=row.get("h3_cell"),
                metadata=dict(row.get("metadata", {})),
            )
            for row in payload["waypoints"]
        ),
        travel_time_s=np.array(payload["travel_time_s"], dtype=np.int64),
        budget_s=int(payload["budget_s"]),
        metadata=dict(payload.get("metadata", {})),
    )


def write_instance_json(instance: Instance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance_to_dict(instance), indent=2, sort_keys=True), encoding="utf-8")


def read_instance_json(path: Path) -> Instance:
    return instance_from_dict(json.loads(path.read_text(encoding="utf-8")))
