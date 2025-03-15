import json
import logging as log
import os
import pandas as pd

from neuraflux.geography import CityEnum
from neuraflux.schemas.agency import (
    AgentConfig,
    AgentControlConfig,
    AgentDataConfig,
    RLConfig,
    SignalTags,
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
    """
    # Define the asset-related configurations
    SIGNALS_INFO = {
        "internal_energy": {
            "initial_value": 0,
            "tags": [SignalTags.STATE.value],
            "temporal_knowledge": 0,
            "min_value": 0,
            "max_value": 500,
            "scalable": True,
        },
        "price": {
            "initial_value": 0,
            "tags": [SignalTags.EXOGENOUS.value],
            "temporal_knowledge": 0,
        },
    }
    CONTROL_POWER_MAPPING = {0: -250, 1: 0, 2: 250}
    INITIAL_STATE_DICT = {
        k: v["initial_value"]
        for k, v in SIGNALS_INFO.items()
        if v["initial_value"] is not None
    }
    ASSET_CONFIG = EnergyStorageConfig(
        control_power_mapping=CONTROL_POWER_MAPPING,
        capacity_kwh=500,
        initial_state_dict=INITIAL_STATE_DICT,
    )

    # Define the agency-related configurations
    AGENT_CONTROL_CONFIG = AgentControlConfig(
        n_controls=1,
        n_trajectory_samples=1,
        trajectory_length=6,
        reinforcement_learning=RLConfig(),
    )

    AGENT_DATA_CONFIG = AgentDataConfig(
        control_power_mapping=CONTROL_POWER_MAPPING,
        signals_info=SIGNALS_INFO,
        tracked_signals=["internal_energy", "price"],
    )
    AGENT_CONFIG = AgentConfig(
        control=AGENT_CONTROL_CONFIG,
        data=AGENT_DATA_CONFIG,
        tariff="ONTARIO_TOU",
        product="SIMPLE_TARIFF_OPT",
    )

    # Define the simulation configuration
    DATA_CONFIG = SimulationDataConfig(base_dir="Data Module")
    TIME_CONFIG = SimulationTimeConfig(
        start_time="2023-01-01_00-00-00",
        end_time="2023-02-1_00-00-00",
        step_size_s=300,
    )
    GEO_CONFIG = SimulationGeographicalConfig(city=CityEnum.TORONTO)

    SIMULATION_CONFIG = SimulationConfig(
        data=DATA_CONFIG,
        directory="Sim6",
        time=TIME_CONFIG,
        geography=GEO_CONFIG,
        agents={"Agent001": AGENT_CONFIG},
        assets={"Agent001": ASSET_CONFIG},
    )

    config_dict = SIMULATION_CONFIG.model_dump()
    with open("config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    """

    with open("config.json", "r") as f:
        config_dict = json.load(f)

    SIMULATION_CONFIG = SimulationConfig.from_custom_dict(config_dict)
    # SIMULATION_CONFIG = SimulationConfig.model_construct(config_dict)

    # Execute the simulation
    simulation = Simulation(SIMULATION_CONFIG)

    # Make sure pandas prints all columns in logs
    pd.set_option("display.max_columns", None)
    simulation.run()
