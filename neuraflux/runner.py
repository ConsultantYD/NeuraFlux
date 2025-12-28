from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuraflux.schemas.simulation import SimulationConfig
from neuraflux.simulation import Simulation


@dataclass(frozen=True)
class SimulationResult:
    directory: str
    summary: dict[str, Any]


def run_simulation(
    config: SimulationConfig,
    *,
    overwrite: bool = False,
    make_unique: bool = True,
    log_level: int | str = "INFO",
    structured_logging: bool = True,
) -> SimulationResult:
    simulation = Simulation(
        config,
        overwrite=overwrite,
        make_unique=make_unique,
        log_level=log_level,
        structured_logging=structured_logging,
    )
    summary = simulation.run()
    return SimulationResult(directory=simulation.directory, summary=summary)
