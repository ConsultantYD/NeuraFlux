"""Energy storage simulation example.

This script builds a :class:`~neuraflux.schemas.simulation.SimulationConfig` in Python and runs
it via :func:`neuraflux.runner.run_simulation`.

Control + learning schedule (constant-config):

- Control selection is disabled for the first ``RL_AFTER_DAYS`` days, so the asset uses its
  default auto-control.
- After the warmup period, the agent switches to ``q_policy`` with ``epsilon=0.0``.
- Optional learning runs (real + simulated) are enabled via ``ENABLE_REAL_LEARNING`` and
  ``ENABLE_SIM_LEARNING`` and follow the cron triggers configured below.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuraflux.geography import CityEnum
from neuraflux.global_variables import DT_FILE_STR_FORMAT, DT_STR_FORMAT, OAT_KEY
from neuraflux.runner import run_simulation
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.tariffs import AvailableTariffsEnum
from neuraflux.schemas.agency import (
    AgentConfig,
    AgentControlConfig,
    AgentDataConfig,
    ControlSelectionConfig,
    RealLearningConfig,
    RLConfig,
    SignalTags,
    SimLearningConfig,
)
from neuraflux.schemas.asset_config import EnergyStorageConfig
from neuraflux.schemas.simulation import (
    SimulationConfig,
    SimulationDataConfig,
    SimulationGeographicalConfig,
    SimulationTimeConfig,
)

# -----------------------------------------------------------------------------
# Constant config (edit these values, then run with Poetry)
# `poetry run python examples/simulations/run_sim_energy_storage.py`
# -----------------------------------------------------------------------------
OUTPUT_ROOT = Path("simulations/examples/energy_storage")
LOG_LEVEL = "INFO"

SEED = 42
DAYS = 7
RL_AFTER_DAYS = 1  # first day uses auto_control, then q_policy

# Enable learning runs (real + simulated) to validate the full RL pipeline/artifacts.
ENABLE_REAL_LEARNING = True
ENABLE_SIM_LEARNING = True

# Optional: point to a prior run dir (or a weather.db file) to avoid network fetches.
WEATHER_DB_SOURCE: str | None = None


def main() -> int:
    """Run the example simulation and print the output directory."""
    run_id = dt.datetime.now(dt.timezone.utc).strftime(DT_FILE_STR_FORMAT)
    output_dir = str(OUTPUT_ROOT / run_id)

    SIGNALS_INFO = {
        "internal_energy": {
            "tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value],
            "temporal_knowledge": (None, 0),
            "min_value": 0,
            "max_value": 100,
            "scalable": True,
        },
        OAT_KEY: {
            "tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value],
            "temporal_knowledge": (None, 0),
            "min_value": -50,
            "max_value": 50,
            "scalable": True,
        },
    }

    CONTROL_POWER_MAPPING = {0: -100.0, 1: 0.0, 2: 100.0}

    ASSET_CONFIG = EnergyStorageConfig(
        control_power_mapping=CONTROL_POWER_MAPPING,
        capacity_kwh=100,
        initial_state_dict={"internal_energy": 50},
    )

    CONTROL_SELECTION_CONFIG = {
        0: ControlSelectionConfig(enabled=False),
        60 * 60 * 24 * int(RL_AFTER_DAYS): ControlSelectionConfig(
            policy="q_policy", policy_kwargs={"epsilon": 0.0}
        ),
    }

    AGENT_CONTROL_CONFIG = AgentControlConfig(
        n_controllers=1,
        rl_config=RLConfig(
            action_size=len(CONTROL_POWER_MAPPING),
            state_signals=[
                k
                for k, v in SIGNALS_INFO.items()
                if SignalTags.RL_STATE.value in v["tags"]
            ],
        ),
        control_selection=CONTROL_SELECTION_CONFIG,
        real_replay_buffer_size=10000,
        real_learning_configs={
            0: RealLearningConfig(enabled=False),
            60 * 60 * 24 * int(RL_AFTER_DAYS): RealLearningConfig(
                enabled=bool(ENABLE_REAL_LEARNING),
                trigger_freq_cron="0 0 * * *"
            ),
        },
        sim_replay_buffer_size=1000,
        sim_learning_configs={
            0: SimLearningConfig(enabled=False),
            60 * 60 * 24 * int(RL_AFTER_DAYS): SimLearningConfig(
                enabled=bool(ENABLE_SIM_LEARNING),
                trigger_freq_cron="0 0 * * *",
                n_samples=20,
                n_traj_per_sample=1,
                trajectory_len=10,
                policy="q_policy",
                policy_kwargs={"epsilon": 0.5},
            ),
        },
    )

    AGENT_DATA_CONFIG = AgentDataConfig(
        control_power_mapping=CONTROL_POWER_MAPPING,
        signals_info=SIGNALS_INFO,
        tracked_signals=list(SIGNALS_INFO.keys()),
        memory_dump_freq_cron="55 23 * * *",
    )

    AGENT_CONFIG = AgentConfig(
        control=AGENT_CONTROL_CONFIG,
        data=AGENT_DATA_CONFIG,
        tariff=AvailableTariffsEnum.ONTARIO_GEN_TOU.name,
        product=AvailableProductsEnum.SIMPLE_TARIFF_OPT.name,
    )

    start_dt = dt.datetime(2023, 1, 1, 0, 0, 0)
    end_dt = start_dt + dt.timedelta(days=int(DAYS))
    TIME_CONFIG = SimulationTimeConfig(
        start_time=start_dt.strftime(DT_STR_FORMAT),
        end_time=end_dt.strftime(DT_STR_FORMAT),
        step_size_s=300,
    )

    GEO_CONFIG = SimulationGeographicalConfig(city=CityEnum.TORONTO)
    DATA_CONFIG = SimulationDataConfig(weather_db_source=WEATHER_DB_SOURCE)

    SIM_CONFIG = SimulationConfig(
        data=DATA_CONFIG,
        directory=output_dir,
        seed=int(SEED),
        time=TIME_CONFIG,
        geography=GEO_CONFIG,
        agents={"Agent001": AGENT_CONFIG},
        assets={"Agent001": ASSET_CONFIG},
    )

    config = SIM_CONFIG
    result = run_simulation(config, log_level=LOG_LEVEL)

    print(f"Simulation status: {result.summary.get('status')}")
    print(f"Artifacts directory: {result.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
