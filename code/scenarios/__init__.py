"""Explicit registry for simulation scenarios available to the workshop."""

from __future__ import annotations

from .base import SimulationScenario
from .pick_place import SCENARIO as SO100_PICK_PLACE

_SCENARIOS: dict[str, SimulationScenario] = {}


def register_scenario(scenario: SimulationScenario) -> None:
    """Register one scenario and reject duplicate public names."""

    if scenario.name in _SCENARIOS:
        raise ValueError(f"Scenario {scenario.name!r} is already registered.")
    _SCENARIOS[scenario.name] = scenario


def get_scenario(name: str) -> SimulationScenario:
    """Return one registered scenario or raise with the available names."""

    try:
        return _SCENARIOS[name]
    except KeyError as error:
        raise KeyError(
            f"Unknown scenario {name!r}. Available: {', '.join(list_scenarios())}"
        ) from error


def list_scenarios() -> tuple[str, ...]:
    """Return registered scenario names in deterministic order."""

    return tuple(sorted(_SCENARIOS))


register_scenario(SO100_PICK_PLACE)

__all__ = [
    "SimulationScenario",
    "get_scenario",
    "list_scenarios",
    "register_scenario",
]
