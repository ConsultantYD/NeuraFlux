from __future__ import annotations

__all__ = ["Simulation", "SimulationConfig", "SimulationResult", "run_simulation"]


def __getattr__(name: str):
    if name == "Simulation":
        from neuraflux.simulation import Simulation

        return Simulation
    if name == "SimulationConfig":
        from neuraflux.schemas.simulation import SimulationConfig

        return SimulationConfig
    if name == "SimulationResult":
        from neuraflux.runner import SimulationResult

        return SimulationResult
    if name == "run_simulation":
        from neuraflux.runner import run_simulation

        return run_simulation
    raise AttributeError(f"module 'neuraflux' has no attribute {name!r}")
