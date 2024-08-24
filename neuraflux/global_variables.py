from neuraflux.geography import CityEnum

TIMESTAMP_KEY = "timestamp"
DT_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"

FILE_SCALING = "scaling_info"
FILE_REAL_REPLAY_BUFFER = "real_replay_buffer"
FILE_SIM_REPLAY_BUFFER = "sim_replay_buffer"

TABLE_SIGNALS = "asset_signals"
TABLE_SIGNALS_SHADOW = "shadow_asset_signals"
TABLE_CONTROLS = "agent_controls"
TABLE_WEATHER = "weather"
TABLE_DQN_TRAINING = "ddqn_training"

ENERGY_KEY = "energy"
CUMULATIVE_ENERGY_KEY = "cumulative_energy"
POWER_KEY = "power"
OAT_KEY = "outside_air_temperature"

CONTROL_KEY = "control"
REWARD_KEY = "reward"
DONE_KEY = "done"

TARIFF_KEY = "tariff_$"
PRICE_KEY = "price_$"

PRODUCTION_KEY = "prod"

WEATHER_DB_NAME = "weather.db"

# LOGGING
LOGGING_DB_NAME = "logs.db"
LOG_TIMESTAMP_KEY = "timestamp"
LOG_MODULE_KEY = "module"
LOG_LEVEL_KEY = "log level"
LOG_SIM_T_KEY = "simulation_time"
LOG_ENTITY_KEY = "entity"
LOG_METHOD_KEY = "method"
LOG_MESSAGE_KEY = "message"

# GEOGRAPHY
CITIES_INFO = {
    CityEnum.NEW_YORK: {"lat": 40.7128, "lon": -74.0060, "alt": 10},
    CityEnum.LOS_ANGELES: {"lat": 34.0522, "lon": -118.2437, "alt": 71},
    CityEnum.CHICAGO: {"lat": 41.8781, "lon": -87.6298, "alt": 176},
    CityEnum.TORONTO: {"lat": 43.6510, "lon": -79.3470, "alt": 76},
    CityEnum.MEXICO_CITY: {"lat": 19.4326, "lon": -99.1332, "alt": 2240},
    CityEnum.HOUSTON: {"lat": 29.7604, "lon": -95.3698, "alt": 12},
    CityEnum.VANCOUVER: {"lat": 49.2827, "lon": -123.1207, "alt": 70},
}
