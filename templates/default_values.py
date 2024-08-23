# -----------------------------------------------------------------
# - BUILDING ASSET
# -----------------------------------------------------------------
BUILDING_DEF_TRACKED_SIGNALS = [
    "temperature_1",
    "temperature_2",
    "temperature_3",
    "hvac_1",
    "hvac_2",
    "hvac_3",
    "cool_setpoint",
    "heat_setpoint",
    "occupancy",
]
BUILDING_DEF_SIGNAL_INFO = {
    "temperature_1": {
        "tags": ["X"],
        "min_value": 10,
        "max_value": 40,
        "scalable": True,
        "source": "asset",
    },
    "temperature_2": {
        "tags": ["X"],
        "min_value": 10,
        "max_value": 40,
        "scalable": True,
        "source": "asset",
    },
    "temperature_3": {
        "tags": ["X"],
        "min_value": 10,
        "max_value": 40,
        "scalable": True,
        "source": "asset",
    },
    "hvac_1": {"tags": ["U"]},
    "hvac_2": {"tags": ["U"]},
    "hvac_3": {"tags": ["U"]},
    "cool_setpoint": {"tags": ["W"]},
    "heat_setpoint": {"tags": ["W"]},
    "occupancy": {"tags": ["W"]},
}

BUILDING_DEF_CMP = {
    0: 20,  # Cooling Stage 2
    1: 10,  # Cooling Stage 1
    2: 0,  # Control Off
    3: 10,  # Heating Stage 1
    4: 20,  # Heating Stage 2
}

BUILDING_DEF_INITIAL_STATE_DICT = {"temperature": [21.0, 21.0, 21.0], "hvac": [0, 0, 0]}

# -----------------------------------------------------------------
# - ENERGY STORAGE ASSET
# -----------------------------------------------------------------
ENERGY_STORAGE_DEF_TRACKED_SIGNALS = [
    "internal_energy",
    "market_price_t",
    "market_price_t+1",
    "market_price_t+2",
    "market_price_t+3",
    "market_price_t+4",
    "market_price_t+5",
    "market_price_t+6",
]

ENERGY_STORAGE_DEF_SIGNAL_INFO = {
    "internal_energy": {
        "tags": ["X", "S"],
        "min_value": 0,
        "max_value": 500,
        "scalable": True,
        "source": "asset",
    },
    "market_price_t": {
        "tags": ["W", "S"],
        "source": "product",
    },
    "market_price_t+1": {
        "tags": ["W", "S"],
        "source": "product",
    },
    "market_price_t+2": {
        "tags": ["W", "S"],
        "source": "product",
    },
    "market_price_t+3": {
        "tags": ["W", "S"],
        "source": "product",
    },
    "market_price_t+4": {
        "tags": ["W", "S"],
        "source": "product",
    },
    "market_price_t+5": {
        "tags": ["W", "S"],
        "source": "product",
    },
    "market_price_t+6": {
        "tags": ["W", "S"],
        "source": "product",
    },
}

ENERGY_STORAGE_DEF_CMP = {
    0: -250,  # Sell
    1: 0,  # Do nothing
    2: 250,  # Buy
}

ENERGY_STORAGE_DEF_INITIAL_STATE_DICT = {"internal_energy": 250.0}
