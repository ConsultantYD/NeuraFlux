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
PRELOADED_SHADOW_ASSET_DF_KEY = "preloaded_shadow_asset_df"

CONTROL_MODULE_KEY = "control_module"
DATA_MODULE_KEY = "data_module"

SELECTED_SIM_CONFIG_KEY = "selected_sim_config"
SELECTED_SIM_NAME_KEY = "sim_name"
SELECTED_SIM_DIR_KEY = "sim_dir"
SELECTED_SIM_SUMMARY_KEY = "sim_summary"

# Constants for UI
ASSET_DESCRIPTION_DICT: dict[str, str] = {
    "Commercial Building": "From single-story structures to multi-story complexes, hosting offices, retail spaces, and other establishments. Key controllable systems typically include HVAC, lighting, and behind-the-meter assets. The primary stakeholders impacted and served by this solution are tenants, facility managers, and building owners.",
    "Electric Vehicle": "Powered by electric motors and rechargeable batteries, electric vehicles provide an eco-friendly alternative to traditional gasoline-powered cars, offering reduced emissions and sustainable transportation.",
    "EV Charger": "Devices designed to supply electrical energy for recharging electric vehicle batteries, these chargers come in various types and speeds to accommodate different charging needs.",
    "Energy Storage": "Systems that capture and store energy for later use provide a reliable supply of electricity during power outages or when demand exceeds supply, ensuring consistent energy availability.",
    "Residential Building": "Designed to provide comfort and shelter, residential buildings range from single-family homes to apartment complexes, serving as housing structures for individuals or families.",
}

SANITIZED_PRODUCTS_MAPPING: dict[str, str] = {
    "Building HVAC Optimization": "Energy Efficiency",
    "Demand Response": "Demand Response",
    "Dynamic Pricing (CAISO)": "Arbitrage",
    "Tariff Optimization": "Tariff Optimization",
    "Tariff, GHG, and DR Optimization": "Decarbonization",
    "DYNAMIC_PRICING": "Arbitrage",
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
