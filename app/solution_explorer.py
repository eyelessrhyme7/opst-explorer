from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from opst.instance_generation import read_instance_json  # noqa: E402

RESULTS = ROOT / "results/raw/exact_suite_results.csv"
SOLUTIONS = ROOT / "results/raw/exact_suite_solutions.jsonl"
MANIFEST = ROOT / "data/instances/exact_suite/manifest.csv"
PAIRED = ROOT / "results/tables/exact_suite_paired_improvement_rows.csv"

COLORS = {
    "exact_subset_dp": [255, 0, 0],
    "split_only_grid": [0, 120, 255],
    "split_compact_grid": [0, 255, 60],
    "cheapest_insertion": [255, 180, 0],
    "service_density_then_route": [255, 0, 255],
    "best_single": [255, 255, 255],
}

COLOR_HEX = {
    algorithm: f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" for algorithm, rgb in COLORS.items()
}


def offset_path(
    start: dict[str, object], end: dict[str, object], offset_m: float
) -> list[list[float]]:
    """Offset a visual segment so identical routes remain distinguishable."""
    lat_scale = 110_540.0
    avg_lat = (float(start["lat"]) + float(end["lat"])) / 2.0
    lon_scale = 111_320.0 * max(0.1, abs(math.cos(math.radians(avg_lat))))
    dx = (float(end["lon"]) - float(start["lon"])) * lon_scale
    dy = (float(end["lat"]) - float(start["lat"])) * lat_scale
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return [[float(start["lon"]), float(start["lat"])], [float(end["lon"]), float(end["lat"])]]
    nx = -dy / length
    ny = dx / length
    dlon = nx * offset_m / lon_scale
    dlat = ny * offset_m / lat_scale
    return [
        [float(start["lon"]) + dlon, float(start["lat"]) + dlat],
        [float(end["lon"]) + dlon, float(end["lat"]) + dlat],
    ]


def depot_square(stations: pd.DataFrame, size_m: float = 45.0) -> list[dict[str, object]]:
    depot = stations[stations["role"] == "depot"].iloc[0].to_dict()
    lat_scale = 110_540.0
    lon_scale = 111_320.0 * max(0.1, abs(math.cos(math.radians(float(depot["lat"])))))
    half_lon = (size_m / 2.0) / lon_scale
    half_lat = (size_m / 2.0) / lat_scale
    lon = float(depot["lon"])
    lat = float(depot["lat"])
    depot["polygon"] = [
        [lon - half_lon, lat - half_lat],
        [lon + half_lon, lat - half_lat],
        [lon + half_lon, lat + half_lat],
        [lon - half_lon, lat + half_lat],
    ]
    return [depot]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, str], pd.DataFrame]:
    results = pd.read_csv(RESULTS)
    records = {}
    with SOLUTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[f"{record['instance_id']}::{record['algorithm']}"] = record
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        paths = {
            row["instance_id"]: row["path"].replace("\\", "/") for row in csv.DictReader(handle)
        }
    paired = pd.read_csv(PAIRED)
    return results, records, paths, paired


def station_table(instance_id: str, paths: dict[str, str]) -> pd.DataFrame:
    instance = read_instance_json(ROOT / paths[instance_id])
    rows = [
        {
            "station_id": instance.depot_id.removeprefix("station_"),
            "name": "DEPOT",
            "lat": instance.depot_lat,
            "lon": instance.depot_lon,
            "reward": 0,
            "service_time_s": 0,
            "h3_cell": "",
            "role": "depot",
        }
    ]
    for waypoint in instance.waypoints:
        rows.append(
            {
                "station_id": str(waypoint.metadata.get("station_id")),
                "name": waypoint.metadata.get("name", waypoint.waypoint_id),
                "lat": waypoint.lat,
                "lon": waypoint.lon,
                "reward": waypoint.reward,
                "service_time_s": waypoint.service_time_s,
                "h3_cell": waypoint.h3_cell or "",
                "role": "candidate",
            }
        )
    return pd.DataFrame(rows)


def route_layers(
    instance_id: str,
    algorithms: list[str],
    records: dict[str, dict[str, object]],
    stations: pd.DataFrame,
) -> list[pdk.Layer]:
    station_lookup = stations.set_index("station_id").to_dict("index")
    layers: list[pdk.Layer] = []
    offset_step_m = 14.0
    algorithm_offsets = {
        algorithm: (index - (len(algorithms) - 1) / 2.0) * offset_step_m
        for index, algorithm in enumerate(algorithms)
    }
    for algorithm in algorithms:
        record = records.get(f"{instance_id}::{algorithm}")
        if not record:
            continue
        route = [str(station_id) for station_id in record["route_station_ids"]]
        segments = []
        visit_points = []
        for order, station_id in enumerate(route):
            station = station_lookup[station_id]
            if station["role"] != "depot":
                visit_points.append(
                    {
                        **station,
                        "station_id": station_id,
                        "visit_order": order,
                        "algorithm": algorithm,
                        "color": COLORS[algorithm],
                    }
                )
        for a, b in zip(route, route[1:], strict=False):
            start = station_lookup[a]
            end = station_lookup[b]
            segments.append(
                {
                    "path": offset_path(start, end, algorithm_offsets[algorithm]),
                    "algorithm": algorithm,
                    "color": COLORS[algorithm],
                    "visual_offset_m": algorithm_offsets[algorithm],
                }
            )
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=segments,
                get_path="path",
                get_color="color",
                width_min_pixels=7,
                pickable=True,
            )
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=visit_points,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=28,
                pickable=True,
            )
        )
    return layers


def shortcut_instance(
    results: pd.DataFrame, paired: pd.DataFrame, choice: str, algorithm: str
) -> str:
    if choice == "Worst case of selected algorithm":
        rows = results[results["algorithm"] == algorithm].sort_values("reward_ratio")
        return str(rows.iloc[0]["instance_id"])
    if choice == "Largest compactness improvement":
        rows = paired[paired["comparison"] == "split_only_to_split_compact"].sort_values(
            "delta_reward_ratio", ascending=False
        )
        return str(rows.iloc[0]["instance_id"])
    if choice == "Largest split-only failure":
        rows = results[results["algorithm"] == "split_only_grid"].sort_values("reward_ratio")
        return str(rows.iloc[0]["instance_id"])
    if choice == "Representative compact case":
        rows = results[
            (results["regime"] == "compact") & (results["algorithm"] == "exact_subset_dp")
        ]
        return str(rows.iloc[len(rows) // 2]["instance_id"])
    if choice == "Representative outlier case":
        rows = results[
            (results["regime"] == "outlier") & (results["algorithm"] == "exact_subset_dp")
        ]
        return str(rows.iloc[len(rows) // 2]["instance_id"])
    return str(results.iloc[0]["instance_id"])


def main() -> None:
    st.set_page_config(page_title="OP-ST Solution Explorer", layout="wide")
    st.title("OP-ST Exact-Suite Solution Explorer")
    st.caption(
        "Optimization uses the symmetrized OSM travel-time matrix. "
        "Displayed route segments connect station coordinates for visual comparison; "
        "full OSM road geometry is not stored per solution."
    )
    results, records, paths, paired = load_data()
    query = st.query_params
    quick_case_options = [
        "Manual selection",
        "Worst case of selected algorithm",
        "Largest compactness improvement",
        "Largest split-only failure",
        "Representative compact case",
        "Representative outlier case",
    ]

    with st.sidebar:
        st.header("Filters")
        n_values = st.multiselect(
            "n", sorted(results["n"].unique()), default=sorted(results["n"].unique())
        )
        regimes = st.multiselect(
            "regime", sorted(results["regime"].unique()), default=sorted(results["regime"].unique())
        )
        service_scales = st.multiselect(
            "service scale",
            sorted(results["service_scale"].unique()),
            default=sorted(results["service_scale"].unique()),
        )
        budgets = st.multiselect(
            "budget",
            sorted(results["budget_tightness"].unique()),
            default=sorted(results["budget_tightness"].unique()),
        )
        seeds = st.multiselect(
            "seed", sorted(results["seed"].unique()), default=sorted(results["seed"].unique())
        )
        default_quick_case = query.get("quick_case", "Manual selection")
        if default_quick_case not in quick_case_options:
            default_quick_case = "Manual selection"
        shortcut = st.selectbox(
            "Quick case",
            quick_case_options,
            index=quick_case_options.index(default_quick_case),
        )
        shortcut_algorithm_options = sorted(results["algorithm"].unique())
        default_shortcut_algorithm = query.get("shortcut_algorithm", shortcut_algorithm_options[0])
        if default_shortcut_algorithm not in shortcut_algorithm_options:
            default_shortcut_algorithm = shortcut_algorithm_options[0]
        shortcut_algorithm = st.selectbox(
            "Algorithm for worst-case shortcut",
            shortcut_algorithm_options,
            index=shortcut_algorithm_options.index(default_shortcut_algorithm),
        )

    filtered = results[
        results["n"].isin(n_values)
        & results["regime"].isin(regimes)
        & results["service_scale"].isin(service_scales)
        & results["budget_tightness"].isin(budgets)
        & results["seed"].isin(seeds)
    ]
    instance_ids = sorted(filtered["instance_id"].unique())
    if shortcut == "Manual selection":
        default_instance_id = query.get("instance_id", instance_ids[0])
        if default_instance_id not in instance_ids:
            default_instance_id = instance_ids[0]
        instance_id = st.selectbox(
            "Instance", instance_ids, index=instance_ids.index(default_instance_id)
        )
    else:
        instance_id = shortcut_instance(results, paired, shortcut, shortcut_algorithm)
        st.info(f"Shortcut selected: `{instance_id}`")

    default_algorithms = ["exact_subset_dp", "split_only_grid", "split_compact_grid"]
    requested_algorithms = query.get("algorithms", "")
    if requested_algorithms:
        default_algorithms = [
            algorithm
            for algorithm in requested_algorithms.split(",")
            if algorithm in set(results["algorithm"].unique())
        ] or default_algorithms
    algorithms = st.multiselect(
        "Algorithms",
        sorted(results["algorithm"].unique()),
        default=default_algorithms,
    )
    st.markdown("### Route Legend")
    st.caption(
        "Routes are slightly offset in parallel for display only, "
        "so identical paths do not hide each other."
    )
    legend_html = " ".join(
        f"<span style='display:inline-block;margin:0 14px 8px 0;'>"
        f"<span style='display:inline-block;width:18px;height:18px;"
        f"background:{COLOR_HEX[algorithm]};"
        f"border:1px solid #111;margin-right:6px;vertical-align:middle;'></span>"
        f"<code>{algorithm}</code></span>"
        for algorithm in algorithms
    )
    st.markdown(legend_html, unsafe_allow_html=True)
    stations = station_table(instance_id, paths)
    candidate_layer = pdk.Layer(
        "ScatterplotLayer",
        data=stations[stations["role"] == "candidate"],
        get_position="[lon, lat]",
        get_fill_color=[170, 170, 170],
        get_radius=18,
        pickable=True,
    )
    depot_layer = pdk.Layer(
        "PolygonLayer",
        data=depot_square(stations),
        get_polygon="polygon",
        get_fill_color=[0, 255, 255],
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
        pickable=True,
    )
    layers = [
        candidate_layer,
        depot_layer,
        *route_layers(instance_id, algorithms, records, stations),
    ]
    view_state = pdk.ViewState(
        latitude=float(stations["lat"].mean()),
        longitude=float(stations["lon"].mean()),
        zoom=13,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "text": (
                    "{algorithm}\n{name}\nstation {station_id}\nreward {reward}\n"
                    "service {service_time_s}s\nH3 {h3_cell}\nvisit {visit_order}"
                )
            },
        ),
        width="stretch",
    )
    st.subheader("Selected Algorithm Metrics")
    metrics = results[
        (results["instance_id"] == instance_id) & (results["algorithm"].isin(algorithms))
    ]
    st.dataframe(
        metrics[
            [
                "algorithm",
                "reward",
                "reward_ratio",
                "travel_time_s",
                "service_time_s",
                "total_time_s",
                "budget_s",
                "selected_count",
                "runtime_ms",
                "chosen_alpha",
                "chosen_pool_type",
            ]
        ],
        width="stretch",
    )


if __name__ == "__main__":
    main()
