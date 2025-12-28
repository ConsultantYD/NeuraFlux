# Simulation Examples

These examples build a `SimulationConfig` in Python and run it.

Run from the repo root:

- `poetry run python examples/simulations/energy_storage/run_energy_storage.py`
- `poetry run python examples/simulations/commercial_building/run_commercial_building.py`

Each run writes artifacts under `simulations/examples/...` (e.g. `config.json`, `sim_summary.json`, `logs.db`, `weather.db`, agent parquet data, and `agent.pkl`).
