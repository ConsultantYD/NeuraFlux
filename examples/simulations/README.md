# Simulation Examples

These examples build a `SimulationConfig` in Python and run it.

Run from the repo root:

- `poetry run python examples/simulations/run_sim_energy_storage.py`
- `poetry run python examples/simulations/run_sim_commercial_building.py`

Each run writes artifacts under `simulations/examples/...` (e.g. `config.json`, `sim_summary.json`, `logs.db`, `weather.db`, agent parquet data, and `agent.pkl`).

Common top-level artifacts:
- `config.json`: full simulation config
- `sim_summary.json`: run status + runtime metadata
- `time_ref.json`: time reference used by the simulation
- `metrics.json`: rollups for real vs shadow/baseline trajectories
- `logs.db`: structured logs (sqlite), when enabled
- `weather.db`: cached weather data

Per-agent artifacts (under `.../<run_id>/<agent_uid>/`):
- `agent.pkl`: pickled agent snapshot
- `training_summary.json`: counters/timestamps for real/sim training runs
- `data/`: partitioned parquet of timestep records
- `sim_data/`: partitioned parquet of simulated trajectories (when sim learning is enabled)
