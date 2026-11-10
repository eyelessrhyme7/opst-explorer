from __future__ import annotations

from dataclasses import dataclass

from opst.models import Instance, Solution


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    reward: int
    travel_time_s: int
    service_time_s: int
    total_time_s: int
    feasible: bool

    @property
    def ok(self) -> bool:
        return not self.errors


def route_totals(instance: Instance, route_indices: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Return reward, travel, service, and total seconds for a closed route."""

    travel = 0
    for a, b in zip(route_indices, route_indices[1:]):
        travel += int(instance.travel_time_s[a, b])
    selected = tuple(index for index in route_indices[1:-1])
    reward = sum(instance.waypoints[index - 1].reward for index in selected)
    service = sum(instance.waypoints[index - 1].service_time_s for index in selected)
    return reward, travel, service, travel + service


def make_solution(
    instance: Instance,
    algorithm: str,
    route_indices: tuple[int, ...],
    runtime_ms: float,
    parameters: dict[str, object] | None = None,
    status: str = "ok",
) -> Solution:
    """Build a Solution from a route using independently recomputed totals."""

    reward, travel, service, total = route_totals(instance, route_indices)
    selected = tuple(route_indices[1:-1])
    return Solution(
        instance_id=instance.instance_id,
        algorithm=algorithm,
        route_indices=route_indices,
        selected_indices=selected,
        reward=reward,
        travel_time_s=travel,
        service_time_s=service,
        total_time_s=total,
        feasible=total <= instance.budget_s,
        runtime_ms=runtime_ms,
        parameters={} if parameters is None else parameters,
        status=status,
    )


def validate_solution(
    instance: Instance,
    solution: Solution,
    *,
    strict: bool = False,
) -> ValidationResult:
    """Validate a solution against the canonical instance schema."""

    errors: list[str] = []
    route = solution.route_indices
    n = len(instance.waypoints)

    if solution.instance_id != instance.instance_id:
        errors.append("solution instance_id does not match instance")
    if len(route) < 2:
        errors.append("route must contain at least depot start and depot end")
    if route and route[0] != 0:
        errors.append("route must start at depot index 0")
    if route and route[-1] != 0:
        errors.append("route must end at depot index 0")

    for index in route:
        if index < 0 or index > n:
            errors.append(f"route contains invalid index {index}")

    intermediate = tuple(route[1:-1]) if len(route) >= 2 else ()
    if any(index == 0 for index in intermediate):
        errors.append("depot may only appear at the route endpoints")
    if len(set(intermediate)) != len(intermediate):
        errors.append("route visits a station more than once")
    if tuple(sorted(solution.selected_indices)) != tuple(sorted(intermediate)):
        errors.append("selected_indices do not match route nodes")

    if all(0 <= index <= n for index in route):
        reward, travel, service, total = route_totals(instance, route)
    else:
        reward = travel = service = total = 0

    feasible = total <= instance.budget_s
    if solution.reward != reward:
        errors.append(f"reward field {solution.reward} != recomputed {reward}")
    if solution.travel_time_s != travel:
        errors.append(f"travel_time_s field {solution.travel_time_s} != recomputed {travel}")
    if solution.service_time_s != service:
        errors.append(f"service_time_s field {solution.service_time_s} != recomputed {service}")
    if solution.total_time_s != total:
        errors.append(f"total_time_s field {solution.total_time_s} != recomputed {total}")
    if solution.feasible != feasible:
        errors.append(f"feasible field {solution.feasible} != recomputed {feasible}")
    if not feasible:
        errors.append("route exceeds budget")

    result = ValidationResult(
        errors=tuple(errors),
        reward=reward,
        travel_time_s=travel,
        service_time_s=service,
        total_time_s=total,
        feasible=feasible,
    )
    if strict and not result.ok:
        raise ValueError("; ".join(result.errors))
    return result
