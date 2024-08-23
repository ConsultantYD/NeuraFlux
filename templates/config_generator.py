import json

from default_values import (
    ENERGY_STORAGE_DEF_INITIAL_STATE_DICT,
    ENERGY_STORAGE_DEF_CMP,
    ENERGY_STORAGE_DEF_SIGNAL_INFO,
    ENERGY_STORAGE_DEF_TRACKED_SIGNALS,
)
from neuraflux.geography import CityEnum
from neuraflux.schemas.agency import (
    AgentConfig,
    AgentControlConfig,
    AgentDataConfig,
    RLConfig,
)
from neuraflux.schemas.asset_config import BuildingConfig, EnergyStorageConfig
from neuraflux.schemas.simulation import (
    SimulationConfig,
    SimulationGeographicalConfig,
    SimulationTimeConfig,
)

# -----------------------------------------------------------------
# TIME CONFIGURATION SUBSECTION
# -----------------------------------------------------------------
TIME_CONFIG = SimulationTimeConfig(
    start_time="2023-01-01_00-00-00",
    end_time="2023-06-1_00-00-00",
    step_size_s=300,
)


# -----------------------------------------------------------------
# GEOGRAPHY CONFIGURATION SECTION
# -----------------------------------------------------------------
cities_list = CityEnum.get_all_available_city_values()
city_str = cities_list[0]
GEO_CONFIG = SimulationGeographicalConfig(city=CityEnum.from_string(city_str))


# -----------------------------------------------------------------
# AGENT CONFIGURATION SECTION
# -----------------------------------------------------------------
# AGENT CONTROL
AGENT_CONTROL_CONFIG = AgentControlConfig(
    n_controllers=1,
    n_trajectory_samples=1,
    trajectory_length=6,
    reinforcement_learning=RLConfig(
        history_length=6,  # Number of time steps to consider
        add_hourly_time_features_to_state=True,
        add_daily_time_features_to_state=True,
        add_weekly_time_features_to_state=False,
        add_monthly_time_features_to_state=False,
        discount_factor=0.99,  # Discount factor for future rewards
        n_target_updates=10,  # Number of major target network update loops
        n_sampling_iters=5,  # Number of times to fit the model before target update
        experience_sampling_size=500,  # n of exp from PER at each sampling iter
        learning_rate=0.00025,  # Gradient descent learning rate in fit
        n_fit_epochs=1,  # Number of tensorflow epochs to fit the DQN targets
        tf_batch_size=8,  # Tensorflow training batch size
        replay_buffer_size=20000,
        prioritized_replay_alpha=0.6,
    ),
)

# AGENT DATA
AGENT_DATA_CONFIG = AgentDataConfig(
    control_power_mapping=ENERGY_STORAGE_DEF_CMP,
    tracked_signals=ENERGY_STORAGE_DEF_TRACKED_SIGNALS,
    signals_info=ENERGY_STORAGE_DEF_SIGNAL_INFO,
)

# GLOBAL AGENT CONFIG
TARIFF_STR = "NO_TARIFF"
PRODUCT_STR = "DYNAMIC_PRICING"
AGENT_CONFIG = AgentConfig(
    asset_metadata={
        "address": "123 Fake St, Anytown, CA",
        "timezone": "America/Toronto",
        "location": (43.7, -79.42, 0.0),
        "owner": "John Doe",
    },
    components_metadata=[],
    control=AGENT_CONTROL_CONFIG,
    data=AGENT_DATA_CONFIG,
    product=PRODUCT_STR,
    tariff=TARIFF_STR,
)

# -----------------------------------------------------------------
# SIMULATION CONFIGURATION SECTION
# -----------------------------------------------------------------
ASSET_CONFIG = EnergyStorageConfig(
    initial_state_dict=ENERGY_STORAGE_DEF_INITIAL_STATE_DICT,
    control_power_mapping=ENERGY_STORAGE_DEF_CMP
)

SIMULATION_CONFIG = SimulationConfig(
    directory="simulations/NewSim1",
    time=TIME_CONFIG,
    geography=GEO_CONFIG,
    agents={"Agent001": AGENT_CONFIG},
    assets={"Agent001": ASSET_CONFIG},
)

# -----------------------------------------------------------------
# VALIDITY AND COMPATIBILITY CHECKS
# -----------------------------------------------------------------

# Configuration from Python
config_json = SIMULATION_CONFIG.model_dump_json(indent=4)
with open("config.json", "w") as f:
    f.write(config_json)

# Configuration from JSON file
loaded_config = json.load(open("config.json"))
loaded_config = SimulationConfig.model_validate(loaded_config)

print(SIMULATION_CONFIG == loaded_config)
print(SIMULATION_CONFIG)
print("-------------------")
print(loaded_config)
