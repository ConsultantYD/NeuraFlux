import json

from neuraflux.global_variables import OAT_KEY
from neuraflux.geography import CityEnum
from neuraflux.schemas.agency import (
    AgentConfig,
    AgentControlConfig,
    AgentDataConfig,
    RLConfig,
    SignalTags,
)
from neuraflux.schemas.asset_config import BuildingConfig
from neuraflux.schemas.simulation import (
    SimulationConfig,
    SimulationDataConfig,
    SimulationGeographicalConfig,
    SimulationTimeConfig,
)
from neuraflux.simulation import Simulation

if __name__ == "__main__":
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
            "tags": [
                SignalTags.OBSERVATION.value,
            ]
        },
        "hvac_2": {
            "tags": [
                SignalTags.OBSERVATION.value,
            ]
        },
        "hvac_3": {
            "tags": [
                SignalTags.OBSERVATION.value,
            ]
        },
        "cool_setpoint": {
            "tags": [
                SignalTags.EXOGENOUS.value,
                SignalTags.RL_STATE.value,
            ]
        },
        "heat_setpoint": {
            "tags": [
                SignalTags.EXOGENOUS.value,
                SignalTags.RL_STATE.value,
            ]
        },
        "occupancy": {
            "tags": [
                SignalTags.EXOGENOUS.value,
                SignalTags.RL_STATE.value,
            ]
        },
    }

    CONTROL_POWER_MAPPING = {
        0: 40,  # Cooling Stage 2
        1: 20,  # Cooling Stage 1
        2: 0,  # Control Off
        3: 20,  # Heating Stage 1
        4: 40,  # Heating Stage 2
    }

    ASSET_CONFIG = BuildingConfig()

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
        tariff="ONTARIO_GEN_TOU",
        product="HVAC_TARIFF_COMFORT",
    )

    # Define the simulation configuration
    DATA_CONFIG = SimulationDataConfig(base_dir="Data Module")
    TIME_CONFIG = SimulationTimeConfig(
        start_time="2023-01-01T00:00:00",
        end_time="2023-01-30T00:00:00",
        step_size_s=300,
    )

    GEO_CONFIG = SimulationGeographicalConfig(city=CityEnum.TORONTO)

    SIMULATION_CONFIG = SimulationConfig(
        data=DATA_CONFIG,
        directory="simulations/simple_validation",
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
