from copy import deepcopy
import datetime as dt
import json

import pandas as pd
import streamlit as st
from neuraflux.geography import CityEnum
from neuraflux.global_variables import CITIES_INFO, DT_STR_FORMAT
from neuraflux.schemas.simulation import SimulationConfig
from neuraflux.schemas.agency import AgentConfig, SignalInfo
from neuraflux.schemas.asset_config import AvailableAssetsEnum
from neuraflux.agency.tariffs import AvailableTariffsEnum
from neuraflux.agency.products import AvailableProductsEnum
from sidebar_utils import generate_sidebar
from agent_utils import show_agent_ui


# def show_agent_ui(agent_uid: str, agent_config: AgentConfig, locked=False):
#     col41, col42 = st.columns(2)

#     st.write("#### 🧵 Data Configuration")
#     signals_df_dict = {
#         "Signal": [s for s in agent_config.data.signals_info.keys()],
#         "Tags": [s.tags for s in agent_config.data.signals_info.values()],
#         "Min": [s.min_value for s in agent_config.data.signals_info.values()],
#         "Max": [s.max_value for s in agent_config.data.signals_info.values()],
#         "Scalable": [s.scalable for s in agent_config.data.signals_info.values()],
#     }
#     signal_df = pd.DataFrame(signals_df_dict)
#     signal_df.set_index("Signal", inplace=True)
#     st.dataframe(signal_df, use_container_width=True)
#     signal_popover = st.popover("💥 Add Signal", use_container_width=True)

#     st.write("#### 🕹️ Control Configuration")
#     st.write("Under Development")

#     st.write("#### 🧳 Product Selection")
#     product_placeholder = st.empty()
#     col71, _, col72, _ = st.columns((6, 1, 5, 1))

#     st.write("#### 💸 Tariff Selection")
#     col81, _ = st.columns((6, 7))
#     with st.form("agent_form_" + agent_uid, border=False):
#         uid = col41.text_input(
#             "Name (UID)", value=agent_uid, disabled=locked, key="uid" + agent_uid
#         )
#         asset_type = col42.selectbox(
#             "Asset Type",
#             AvailableAssetsEnum.list_assets_for_frontend(),
#             key="asset_type" + agent_uid,
#             disabled=locked,
#         )

#         with signal_popover:
#             signal = st.selectbox(
#                 "Available Asset Signals",
#                 ["Internal Energy"],
#                 None,
#                 key="signal" + agent_uid,
#                 disabled=locked,
#             )
#             _, col51, col52, col53, col54, _ = st.columns((5, 10, 10, 10, 10, 2))
#             X_tag = col51.checkbox("X", False, key="X" + agent_uid, disabled=locked)
#             U_tag = col52.checkbox("U", False, key="U" + agent_uid, disabled=locked)
#             W_tag = col53.checkbox("W", False, key="W" + agent_uid, disabled=locked)
#             O_tag = col54.checkbox("O", False, key="O" + agent_uid, disabled=locked)

#             active_signals = []
#             if X_tag:
#                 active_signals.append("X")
#             if U_tag:
#                 active_signals.append("U")
#             if W_tag:
#                 active_signals.append("W")
#             if O_tag:
#                 active_signals.append("O")

#             col61, col62, _, col63 = st.columns(
#                 (12, 12, 4, 12), vertical_alignment="bottom"
#             )

#             min_value_known = col61.toggle(
#                 "Min Value Known", False, key="min_value_known" + agent_uid
#             )
#             min_value = col61.number_input(
#                 "Min Value",
#                 key="min_value" + agent_uid,
#                 disabled=locked or not min_value_known,
#             )
#             max_value_known = col62.toggle(
#                 "Max Value Known", False, key="max_value_known" + agent_uid
#             )
#             max_value = col62.number_input(
#                 "Max Value",
#                 key="max_value" + agent_uid,
#                 disabled=locked or not max_value_known,
#             )
#             scalable = col63.checkbox(
#                 "Scalable", key="scalable" + agent_uid, disabled=locked
#             )

#             if st.button("Add Signal", key="add_signal" + agent_uid, disabled=locked):
#                 agent_config.data.signals_info[signal] = SignalInfo(
#                     tags=active_signals,
#                     min_value=min_value,
#                     max_value=max_value,
#                     scalable=scalable,
#                 )

#         available_products_list = AvailableProductsEnum.list_products()
#         if "prod_placeholder" in st.session_state:
#             current_product_str = st.session_state["prod_placeholder"]
#         else:
#             current_product_str = agent_config.product
        
#         PRODUCTS = {
#             "Arbitrage": "Arbitrage",
#             "Demand Response": 'Demand Response',
#             'Tariff, GHG, and DR Optimization': "Decarbonization",
#             "Energy Efficiency": "Energy Efficiency",
#             "Grid Stability": "Grid Stability",
#             "Load Flexibility": "Load Flexibility",
#             "Power Peaks": 'Building HVAC Optimization',
#             'Tariff Optimization': "Tariff Optimization",
#         }
#         #REVERSE_PRODUCT = {v: k for k, v in PRODUCTS.items()}
#         st.session_state["prod_placeholder"] = col71.selectbox(
#             "Select Product", list(PRODUCTS.keys()), list(PRODUCTS.keys()).index(current_product_str), key="product" + agent_uid, disabled=locked, label_visibility="collapsed"
#         )

#         sanitized_product = PRODUCTS[st.session_state["prod_placeholder"]]
#         if sanitized_product in available_products_list:
#             agent_config.product = sanitized_product

#         with col72.popover(
#             "Show all products",
#         ):
#             st.image("neuraview/media/all_products.png")

#         col71.write("**Description**")
#         col71.write("TBC")
#         col72.image(f"neuraview/media/{sanitized_product}.png", use_column_width=True)

#         current_tariff_str = agent_config.tariff
#         available_tariff_list = AvailableTariffsEnum.list_tariffs()

#         agent_config.tariff = col81.selectbox(
#             "Select Tariff",
#             available_tariff_list,
#             key="tariff" + agent_uid,
#             index=available_tariff_list.index(current_tariff_str),
#             disabled=locked,
#         )

#         if not locked:
#             if st.form_submit_button("Add agent to simulation", type="primary"):
#                 st.session_state["new_simulation_config"].agents[uid] = agent_config
#                 st.rerun()
    # for agent in agents:
    #    with st.expander(f"**{agent.name}**"):
    #        col31, col32 = st.columns((1, 1))
    #        col31.write(f"Latitude: {agent.lat}")
    #        col32.write(f"Longitude: {agent.lon}")
    #        col33, col34 = st.columns((1, 1))
    #        col33.button("Edit")
    #        col34.button("Delete")


def show_simulation_config_ui(sim_config: SimulationConfig, locked: bool = False):
    # -----------------------------------------------------------------
    # TIME CONFIGURATION SUBSECTION
    # -----------------------------------------------------------------
    # Current data
    current_start = dt.datetime.strptime(sim_config.time.start_time, DT_STR_FORMAT)
    current_end = dt.datetime.strptime(sim_config.time.end_time, DT_STR_FORMAT)
    current_step_size = sim_config.time.step_size_s

    # UI
    st.divider()
    st.subheader("🕓 Time")
    with st.form("time_form", border=False):
        col11, col12, col13 = st.columns(3)
        sim_start = col11.date_input(
            "Start Date", value=current_start.date(), disabled=locked
        )
        sim_end = col12.date_input(
            "End Date", value=current_end.date(), disabled=locked
        )
        sim_step_size = col13.number_input(
            "Step Size (s)",
            value=current_step_size,
            step=15,
            format="%d",
            disabled=locked,
        )

        if st.form_submit_button("Save Changes"):
            # Update the simulation config
            sim_config.time.start_time = sim_start.strftime(DT_STR_FORMAT)
            sim_config.time.end_time = sim_end.strftime(DT_STR_FORMAT)
            sim_config.time.step_size_s = sim_step_size
            st.rerun()

    # -----------------------------------------------------------------
    # GEOGRAPHY CONFIGURATION SECTION
    # -----------------------------------------------------------------
    # Current Data
    current_city = sim_config.geography.city

    # UI
    st.divider()
    st.subheader("🌎 Geography")
    col21, _, col22 = st.columns((5, 2, 5), vertical_alignment="bottom")

    available_cities = CityEnum.get_all_available_city_values()
    city = col21.selectbox(
        "Select City",
        available_cities,
        index=available_cities.index(current_city),
        disabled=locked,
    )
    sim_config.geography.city = city
    st.write(
        " 📍 ",
        "**Lat**: ",
        CITIES_INFO[city]["lat"],
        "  **Lon**: ",
        CITIES_INFO[city]["lon"],
        "  **Alt**: ",
        CITIES_INFO[city]["alt"],
        "m",
    )

    if col22.toggle("Show Map", True):
        st.map(pd.DataFrame(CITIES_INFO[city], index=[0]))

    # -----------------------------------------------------------------
    # NETWORK SECTION
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("⚡️ Network Topology")
    col31, _ = st.columns((5, 7))
    col31.info("Under Development")

    # -----------------------------------------------------------------
    # AGENTS SECTION
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("👤 Agents")

    st.write("##### Overview")
    agent_uid_list = list(sim_config.agents.keys())
    agent_summary_dict = {
        "Agent": agent_uid_list,
        "Asset Type": [sim_config.assets[uid].asset_type for uid in agent_uid_list],
        "Product": [sim_config.agents[uid].product for uid in agent_uid_list],
        "Tariff": [sim_config.agents[uid].tariff for uid in agent_uid_list],
        "Nº Controllers": [
            sim_config.agents[uid].control.n_controllers for uid in agent_uid_list
        ],
    }
    agent_summary_df = pd.DataFrame(agent_summary_dict)
    agent_summary_df.set_index("Agent", inplace=True)
    st.dataframe(agent_summary_df, use_container_width=True)

    st.write("##### Detailed View")

    # Loop over existing simulation agents to show details
    for agent_name, agent_config in st.session_state[
        "new_simulation_config"
    ].agents.items():
        with st.expander(f"♟️ **{agent_name}**"):
            show_agent_ui(agent_name, agent_config, locked=True)

    # Add agent expander
    config_template = list(st.session_state["new_simulation_config"].agents.values())[0]
    with st.expander("**❇️  Add Agent**", icon="➕"):
        show_agent_ui("New Agent", deepcopy(config_template), locked=False)


# Page title and sidebar initialization
st.title("⚙️ Simulation Config")
generate_sidebar()

st.write("#####")
col00, _ = st.columns(2)
sim_name = col00.text_input("Simulation Name", value="sim_1")

# Initialize a new simulation config from templates if necessary
if "new_simulation_config" not in st.session_state:
    loaded_json_config = json.load(open("config.json"))
    new_simulation_config = SimulationConfig.model_validate(loaded_json_config)
    st.session_state["new_simulation_config"] = new_simulation_config

show_simulation_config_ui(st.session_state["new_simulation_config"])
