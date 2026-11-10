from __future__ import annotations

from opst.models import Instance
from opst.validate import route_totals


def insertion_delta_s(instance: Instance, route: tuple[int, ...], node: int, position: int) -> int:
    """Marginal total seconds for inserting ``node`` before ``position``."""

    before = route[position - 1]
    after = route[position]
    service = instance.waypoints[node - 1].service_time_s
    travel_delta = (
        int(instance.travel_time_s[before, node])
        + int(instance.travel_time_s[node, after])
        - int(instance.travel_time_s[before, after])
    )
    return service + travel_delta


def best_insertion(instance: Instance, route: tuple[int, ...], node: int) -> tuple[int, int]:
    """Return the cheapest insertion position and marginal total seconds."""

    candidates = (
        (insertion_delta_s(instance, route, node, position), position)
        for position in range(1, len(route))
    )
    delta, position = min(candidates, key=lambda item: (item[0], item[1]))
    return position, delta


def two_opt(instance: Instance, route: tuple[int, ...]) -> tuple[int, ...]:
    """Deterministic 2-opt on travel time; selected service set is unchanged."""

    best = tuple(route)
    improved = True
    while improved:
        improved = False
        current_travel = route_totals(instance, best)[1]
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = (*best[:i], *reversed(best[i : j + 1]), *best[j + 1 :])
                candidate_travel = route_totals(instance, candidate)[1]
                if candidate_travel < current_travel or (
                    candidate_travel == current_travel and candidate < best
                ):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    return best


def route_selected_by_cheapest_insertion(
    instance: Instance, selected_indices: tuple[int, ...]
) -> tuple[int, ...]:
    """Construct a route through a fixed selected set by cheapest insertion."""

    route = (0, 0)
    uninserted = set(selected_indices)
    while uninserted:
        node, position, delta = min(
            (
                (node, *best_insertion(instance, route, node))
                for node in sorted(uninserted)
            ),
            key=lambda item: (item[2], item[0], item[1]),
        )
        route = (*route[:position], node, *route[position:])
        uninserted.remove(node)
    return two_opt(instance, route)


def cheapest_insertion_route(
    instance: Instance, candidate_indices: tuple[int, ...] | None = None
) -> tuple[int, ...]:
    """Build a feasible route by reward per marginal total time."""

    if candidate_indices is None:
        remaining = set(range(1, len(instance.waypoints) + 1))
    else:
        remaining = set(candidate_indices)
    route = (0, 0)
    while remaining:
        candidates: list[tuple[float, int, int, int, int]] = []
        current_total = route_totals(instance, route)[3]
        for node in sorted(remaining):
            position, delta = best_insertion(instance, route, node)
            if current_total + delta > instance.budget_s:
                continue
            reward = instance.waypoints[node - 1].reward
            score = float("inf") if delta == 0 and reward > 0 else reward / max(delta, 1)
            candidates.append((score, reward, -delta, -node, position))
        if not candidates:
            break
        _, _, _, neg_node, position = max(candidates)
        node = -neg_node
        route = (*route[:position], node, *route[position:])
        remaining.remove(node)
        route = two_opt(instance, route)
    return route


def removal_saving_s(instance: Instance, route: tuple[int, ...], position: int) -> int:
    """Actual total-time saving from removing the route node at ``position``."""

    node = route[position]
    before = route[position - 1]
    after = route[position + 1]
    service = instance.waypoints[node - 1].service_time_s
    travel_saving = (
        int(instance.travel_time_s[before, node])
        + int(instance.travel_time_s[node, after])
        - int(instance.travel_time_s[before, after])
    )
    return service + travel_saving


def repair_route_by_removal(instance: Instance, route: tuple[int, ...]) -> tuple[int, ...]:
    """Remove low reward-per-saved-time stations until the route is feasible."""

    repaired = tuple(route)
    while route_totals(instance, repaired)[3] > instance.budget_s and len(repaired) > 2:
        candidates = []
        for position in range(1, len(repaired) - 1):
            node = repaired[position]
            saved = removal_saving_s(instance, repaired, position)
            reward = instance.waypoints[node - 1].reward
            score = float("inf") if saved <= 0 and reward > 0 else reward / max(saved, 1)
            candidates.append((score, reward, -saved, node, position))
        _, _, _, _, position_to_remove = min(candidates)
        repaired = (*repaired[:position_to_remove], *repaired[position_to_remove + 1 :])
        repaired = two_opt(instance, repaired)
    return repaired


def augment_route_feasibly(
    instance: Instance,
    route: tuple[int, ...],
    candidate_indices: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Greedily insert remaining stations by reward per marginal total time."""

    if candidate_indices is None:
        remaining = set(range(1, len(instance.waypoints) + 1))
    else:
        remaining = set(candidate_indices)
    remaining.difference_update(route[1:-1])
    augmented = tuple(route)
    while remaining:
        current_total = route_totals(instance, augmented)[3]
        candidates: list[tuple[float, int, int, int, int]] = []
        for node in sorted(remaining):
            position, delta = best_insertion(instance, augmented, node)
            if current_total + delta > instance.budget_s:
                continue
            reward = instance.waypoints[node - 1].reward
            score = float("inf") if delta == 0 and reward > 0 else reward / max(delta, 1)
            candidates.append((score, reward, -delta, -node, position))
        if not candidates:
            break
        _, _, _, neg_node, position = max(candidates)
        node = -neg_node
        augmented = (*augmented[:position], node, *augmented[position:])
        remaining.remove(node)
        augmented = two_opt(instance, augmented)
    return augmented
