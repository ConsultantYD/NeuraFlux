from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
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
    RLConfig,
    RLTrainingConfig,
    RealLearningConfig,
    SignalTags,
    SimLearningConfig,
)
from neuraflux.schemas.asset_config import BuildingConfig
from neuraflux.schemas.simulation import (
    SimulationConfig,
    SimulationDataConfig,
    SimulationGeographicalConfig,
    SimulationTimeConfig,
)

# -----------------------------------------------------------------------------
# Constant config (edit these values, then run with Poetry)
# `poetry run python examples/simulations/commercial_building/run_commercial_building.py`
# -----------------------------------------------------------------------------
OUTPUT_ROOT = Path("simulations/examples/commercial_building")
LOG_LEVEL = "INFO"

SEED = 42
DAYS = 7
RL_AFTER_DAYS = 1  # start training after 1 day (still on auto_control)
CONTROL_POLICY_AFTER_DAYS = 2  # switch to learned policy after warmup/training

# Enable learning runs (real + simulated) to validate the full RL pipeline/artifacts.
ENABLE_REAL_LEARNING = True
ENABLE_SIM_LEARNING = True

# Optional: point to a prior run dir (or a weather.db file) to avoid network fetches.
WEATHER_DB_SOURCE: str | None = None


def main() -> int:
    run_id = dt.datetime.now(dt.timezone.utc).strftime(DT_FILE_STR_FORMAT)
    output_dir = str(OUTPUT_ROOT / run_id)

    SIGNALS_INFO = {
        "temperature_1": {
            "tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value],
            "temporal_knowledge": (None, 0),
            "min_value": -50,
            "max_value": 50,
            "scalable": True,
        },
        "temperature_2": {
            "tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value],
            "temporal_knowledge": (None, 0),
            "min_value": -50,
            "max_value": 50,
            "scalable": True,
        },
        "temperature_3": {
            "tags": [SignalTags.STATE.value, SignalTags.RL_STATE.value],
            "temporal_knowledge": (None, 0),
            "min_value": -50,
            "max_value": 50,
            "scalable": True,
        },
        OAT_KEY: {
            "tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value],
            "temporal_knowledge": (None, 0),
            "min_value": -50,
            "max_value": 50,
            "scalable": True,
        },
        "hvac_1": {
            "tags": [SignalTags.OBSERVATION.value, SignalTags.RL_STATE.value],
            "min_value": -2,
            "max_value": 2,
            "scalable": True,
        },
        "hvac_2": {
            "tags": [SignalTags.OBSERVATION.value, SignalTags.RL_STATE.value],
            "min_value": -2,
            "max_value": 2,
            "scalable": True,
        },
        "hvac_3": {
            "tags": [SignalTags.OBSERVATION.value, SignalTags.RL_STATE.value],
            "min_value": -2,
            "max_value": 2,
            "scalable": True,
        },
        "cool_setpoint": {
            "tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value]
        },
        "heat_setpoint": {
            "tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value]
        },
        "occupancy": {"tags": [SignalTags.EXOGENOUS.value, SignalTags.RL_STATE.value]},
    }

    ASSET_CONFIG = BuildingConfig()
    CONTROL_POWER_MAPPING = ASSET_CONFIG.control_power_mapping

    CONTROL_SELECTION_CONFIG = {
        0: ControlSelectionConfig(enabled=False),
        60 * 60 * 24 * int(CONTROL_POLICY_AFTER_DAYS): ControlSelectionConfig(
            policy="hvac_policy", policy_kwargs={"epsilon": 0.0}
        ),
    }

    AGENT_CONTROL_CONFIG = AgentControlConfig(
        n_controllers=3,
        rl_config=RLConfig(
            n_controllers=3,
            action_size=len(CONTROL_POWER_MAPPING),
            state_signals=[
                k
                for k, v in SIGNALS_INFO.items()
                if SignalTags.RL_STATE.value in v["tags"]
            ],
        ),
        control_selection=CONTROL_SELECTION_CONFIG,
        real_replay_buffer_size=20000,
        real_learning_configs={
            0: RealLearningConfig(enabled=False),
            60 * 60 * 24 * int(RL_AFTER_DAYS): RealLearningConfig(
                enabled=bool(ENABLE_REAL_LEARNING),
                trigger_freq_cron="0 0 * * *",
                rl_training_config=RLTrainingConfig(
                    n_target_iterators=2,
                    n_sampling_iters=2,
                    experience_sampling_size=128,
                    n_fit_epochs=1,
                    tf_batch_size=16,
                    learning_rate=5e-4,
                ),
            ),
        },
        sim_replay_buffer_size=5000,
        sim_learning_configs={
            0: SimLearningConfig(enabled=False),
            60 * 60 * 24 * int(RL_AFTER_DAYS): SimLearningConfig(
                enabled=bool(ENABLE_SIM_LEARNING),
                trigger_freq_cron="0 0 * * *",
                n_samples=5,
                n_traj_per_sample=1,
                trajectory_len=8,
                policy="hvac_policy",
                policy_kwargs={"epsilon": 0.4, "comfort_constraint": True},
                rl_training_config=RLTrainingConfig(
                    n_target_iterators=2,
                    n_sampling_iters=2,
                    experience_sampling_size=64,
                    n_fit_epochs=1,
                    tf_batch_size=8,
                    learning_rate=1e-4,
                ),
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
        product=AvailableProductsEnum.HVAC_TARIFF_COMFORT.name,
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

    result = run_simulation(SIM_CONFIG, log_level=LOG_LEVEL)

    print(f"Simulation status: {result.summary.get('status')}")
    print(f"Artifacts directory: {result.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
