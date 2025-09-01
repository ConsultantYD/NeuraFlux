import json

from neuraflux.global_variables import OAT_KEY
from neuraflux.geography import CityEnum
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
from neuraflux.simulation import Simulation

if __name__ == "__main__":
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
    CONTROL_POWER_MAPPING = {0: -100, 1: 0, 2: 100}
    INITIAL_STATE_DICT = {
        "internal_energy": 50,
    }

    ASSET_CONFIG = EnergyStorageConfig(
        control_power_mapping=CONTROL_POWER_MAPPING,
        capacity_kwh=100,
        initial_state_dict=INITIAL_STATE_DICT,
    )

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
        control_selection={
            0: ControlSelectionConfig(enabled=False),
            60 * 60 * 24 * 7: ControlSelectionConfig(
                policy="q_policy", policy_kwargs={"epsilon": 0.0}
            ),
        },
        real_replay_buffer_size=10000,
        real_learning_configs={
            0: RealLearningConfig(enabled=False),
            60 * 60 * 24 * 1: RealLearningConfig(trigger_freq_cron="0 0 * * *"),
        },
        sim_replay_buffer_size=1000,
        sim_learning_configs={
            0: SimLearningConfig(enabled=False),
            60 * 60 * 24 * 1: SimLearningConfig(
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
        tariff="NO_TARIFF",
        product="ERCOT_ARBITRAGE",
    )

    # Define the simulation configuration
    DATA_CONFIG = SimulationDataConfig(base_dir="Data Module")
    TIME_CONFIG = SimulationTimeConfig(
        start_time="2023-01-01T00:00:00",
        end_time="2023-04-01T00:00:00",
        step_size_s=300,
    )

    GEO_CONFIG = SimulationGeographicalConfig(city=CityEnum.TORONTO)

    SIMULATION_CONFIG = SimulationConfig(
        data=DATA_CONFIG,
        directory="simulations/case_study_1",
        time=TIME_CONFIG,
        geography=GEO_CONFIG,
        agents={"Agent001": AGENT_CONFIG},
        assets={"Agent001": ASSET_CONFIG},
    )

    config_dict = SIMULATION_CONFIG.model_dump()
    with open("sim_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    with open("sim_config.json", "r") as f:
        config_dict = json.load(f)

    SIMULATION_CONFIG = SimulationConfig.from_custom_dict(config_dict)
    # SIMULATION_CONFIG = SimulationConfig.model_construct(config_dict)

    # Execute the simulation
    simulation = Simulation(SIMULATION_CONFIG)

    # Make sure pandas prints all columns in logs
    # pd.set_option("display.max_columns", None)
    simulation.run()
