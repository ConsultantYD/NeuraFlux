from neuraflux.schemas.asset_config import (
    AssetComponentEnum,
    AvailableAssetsEnum,
    AssetComponent,
)

# Path, Directories and Filenames
SIMS_ROOT_DIR = "simulations"
SIM_SUMMARY_FILENAME = "sim_summary.json"

# Icon files mapping dict
ASSET_TO_ICON_PATH = {
    "commercial building": "commercial_building.png",
    "electric vehicle": "electric_vehicle.png",
    "energy storage": "energy_storage.png",
    "electric vehicle charger": "ev_charger.png",
    "residential building": "residential_building.png",
    "solar panel": "solar_panel.png",
    "wind turbine": "wind_turbine.png",
}

# Session State Keys
AGENTS_KEY = "agents"
ALL_SIM_AGENTS_LIST_KEY = "all_sim_agents_list"
PRELOADED_AGENTS_LIST_KEY = "preloaded_agents_list"
PRELOADED_AGENTS_KEY = "preloaded_agents"
PRELOADED_AGENTS_DF_KEY = "preloaded_agents_df"

CONTROL_MODULE_KEY = "control_module"
DATA_MODULE_KEY = "data_module"

SELECTED_SIM_CONFIG_KEY = "selected_sim_config"
SELECTED_SIM_NAME_KEY = "sim_name"
SELECTED_SIM_DIR_KEY = "sim_dir"
SELECTED_SIM_SUMMARY_KEY = "sim_summary"

# Constants for UI
ASSET_COMPONENTS_DICT: dict[AvailableAssetsEnum, dict[AssetComponent:int]] = {
    AvailableAssetsEnum.COMMERCIAL_BUILDING: {
        AssetComponentEnum.HVAC_ROOF_TOP_UNIT: 3,
        AssetComponentEnum.LIGHTING: 18,
    },
    AvailableAssetsEnum.ELECTRIC_VEHICLE: {},
    AvailableAssetsEnum.EV_CHARGER: {},
    AvailableAssetsEnum.ENERGY_STORAGE: {},
    AvailableAssetsEnum.RESIDENTIAL_BUILDING: {
        AssetComponentEnum.DISHWASHER: 1,
        AssetComponentEnum.HVAC_BASEBOARD: 4,
        AssetComponentEnum.LIGHTING: 8,
        AssetComponentEnum.OVEN: 1,
        AssetComponentEnum.WATER_HEATER: 1,
    },
    AvailableAssetsEnum.SOLAR_PANEL: {},
    AvailableAssetsEnum.WIND_TURBINE: {},
}

ASSET_DESCRIPTION_DICT: dict[AvailableAssetsEnum, str] = {
    AvailableAssetsEnum.COMMERCIAL_BUILDING: "From single-story structures to multi-story complexes, hosting offices, retail spaces, and other establishments. Key controllable systems typically include HVAC, lighting, and behind-the-meter assets. The primary stakeholders impacted and served by this solution are tenants, facility managers, and building owners.",
    AvailableAssetsEnum.ELECTRIC_VEHICLE: "Powered by electric motors and rechargeable batteries, electric vehicles provide an eco-friendly alternative to traditional gasoline-powered cars, offering reduced emissions and sustainable transportation.",
    AvailableAssetsEnum.EV_CHARGER: "Devices designed to supply electrical energy for recharging electric vehicle batteries, these chargers come in various types and speeds to accommodate different charging needs.",
    AvailableAssetsEnum.ENERGY_STORAGE: "Systems that capture and store energy for later use provide a reliable supply of electricity during power outages or when demand exceeds supply, ensuring consistent energy availability.",
    AvailableAssetsEnum.RESIDENTIAL_BUILDING: "Designed to provide comfort and shelter, residential buildings range from single-family homes to apartment complexes, serving as housing structures for individuals or families.",
    AvailableAssetsEnum.SOLAR_PANEL: "Converting sunlight into electricity, solar panels use photovoltaic cells for renewable energy generation, commonly installed on rooftops of residential and commercial properties.",
    AvailableAssetsEnum.WIND_TURBINE: "Machines that convert the kinetic energy of wind into electrical power, wind turbines consist of blades, a rotor, and a generator to produce renewable energy efficiently.",
}

ASSET_INTERNAL_COMPONENTS_DICT: dict[AvailableAssetsEnum, dict] = {
    AvailableAssetsEnum.COMMERCIAL_BUILDING: {
        "Component": ["Air-Handling Unit (AHU)", "Variable Air-Volume (VAV)", "Lights"],
        "Count": [10, 32, 256],
        "Type": ["HVAC", "HVAC", "Lighting"],
        "Controllable": [True, True, True],
        "% Total Elec. ⚡️": [30, 10, 20],
        "% Total Gas 🔥": [80, 0, 0],
    },
    AvailableAssetsEnum.ELECTRIC_VEHICLE: {},
    AvailableAssetsEnum.EV_CHARGER: {},
    AvailableAssetsEnum.ENERGY_STORAGE: {},
    AvailableAssetsEnum.RESIDENTIAL_BUILDING: {
        "Component": ["Heat Pump", "Baseboard", "Lights"],
        "Count": [2, 6, 8],
        "Type": ["HVAC", "HVAC", "Lighting"],
        "Controllable": [True, True, False],
        "% Total Elec. ⚡️": [20, 50, 10],
        "% Total Gas 🔥": [0, 0, 0],
    },
    AvailableAssetsEnum.SOLAR_PANEL: {},
    AvailableAssetsEnum.WIND_TURBINE: {},
}

PRODUCTS_DESCRIPTION_DICT: dict[str, str] = {
    "Arbitrage": "Utilize fluctuating market prices by strategically buying energy when prices are low and selling or consuming it when prices are high to maximize financial returns.",
    "Demand Response": "Adjust energy consumption or curtail load at specific times based on utility needs to balance grid demand and supply, often in exchange for financial incentives.",
    "Decarbonization": "Reduce greenhouse gas (GHG) emissions by prioritizing energy consumption during periods of clean energy generation, thereby supporting a transition to a lower-carbon energy system.",
    "Energy Efficiency": "Enhance consumption patterns of assets to reduce overall energy usage while maintaining or improving the same level of service or output.",
    "Grid Stability": "Support grid stability by providing rapid adjustments to energy consumption or generation in response to frequency changes, ensuring the balance between supply and demand and maintaining reliable grid operations.",
    "Load Flexibility": "Offer dynamic consumption flexibility to utilities by following a predefined pattern that aligns with utility or partner requirements, aiding grid stability and efficiency.",
    "Power Peaks": "Manage and control peak demand to avoid excessive demand charges in tariffs and improve the load factor, ensuring more efficient energy usage and cost savings.",
    "Tariff Optimization": "Optimize the asset's current tariff rate structure to minimize expenses and maximize financial benefits, including credits or profits, by aligning energy use with the most favorable pricing.",
}
