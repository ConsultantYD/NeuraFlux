import pandas as pd
import datetime as dt
import streamlit as st
from neuraflux.schemas.agency import AgentConfig, SignalInfo, SignalTags
from neuraflux.schemas.asset_config import AvailableAssetsEnum
from neuraflux.agency.tariffs import AvailableTariffsEnum
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.schemas.asset_config import AvailableAssetsEnum
from neuraflux.global_variables import DT_STR_FORMAT
from global_variables import (
    ASSET_TO_ICON_PATH,
    ASSET_DESCRIPTION_DICT,
    ASSET_INTERNAL_COMPONENTS_DICT,
    PRODUCTS_DESCRIPTION_DICT,
    PRELOADED_AGENTS_KEY,
    PRELOADED_AGENTS_DF_KEY,
    SELECTED_SIM_SUMMARY_KEY,
)
from plot_utils import create_tou_heatmap, plotly_hist_plot, plotly_line_chart


def asset_section_ui(agent_uid: str, agent_config: AgentConfig, locked: bool = False):
    col11, col12 = st.columns((3, 4))

    # Initialize necessary variables
    frontend_asset_list = AvailableAssetsEnum.list_assets_for_frontend()
    asset_list = AvailableAssetsEnum.list_assets()
    frontend_to_backend_asset_mapping = {
        k: v for k, v in zip(frontend_asset_list, asset_list)
    }

    # LEFT COLUMN - ASSET TYPE AND OTHER INFORMATIONS
    # Asset Type
    asset_type = col11.selectbox(
        "Asset Type",
        frontend_asset_list,
        key="asset_type" + agent_uid,
        disabled=locked,
        label_visibility="collapsed",
    )
    sanitized_asset_type = frontend_to_backend_asset_mapping[asset_type]

    # Asset DF
    asset_df = pd.DataFrame(
        {
            "Location (lat/lon)": ["43.651070, -79.347015"],
            "Adress": ["123 Street, Toronto (ON)"],
            "Owner": ["John Doe"],
        }
    ).T
    asset_df["Characteristics"] = asset_df.index
    asset_df.columns = ["Value", "Characteristics"]
    col11.dataframe(
        asset_df[["Characteristics", "Value"]],
        use_container_width=True,
        hide_index=True,
    )

    # Installed Maximum Capacity
    max_capacity = max([v for v in agent_config.data.control_power_mapping])
    with col11:
        st.write("**Installed Capacity:**", 250.0, "kW")

    # Asset Description
    col11.write("**Description:** " + ASSET_DESCRIPTION_DICT[sanitized_asset_type])

    # Internal Components
    components_df = pd.DataFrame(ASSET_INTERNAL_COMPONENTS_DICT[sanitized_asset_type])
    if len(components_df):
        col21, col22, _ = st.columns((5, 3, 7), vertical_alignment="center")
        col21.write("##### Internal Components")
        col22.button("View Details", key="view_details" + agent_uid)

        col_config = {
            "Type": st.column_config.ListColumn(),
            "% Total Elec. ⚡️": st.column_config.ProgressColumn(),
            "% Total Gas 🔥": st.column_config.ProgressColumn(),
        }

        st.dataframe(
            components_df,
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
        )

    # RIGHT COLUMN - ASSET ICON
    asset_icon_filename = ASSET_TO_ICON_PATH[
        frontend_to_backend_asset_mapping[asset_type]
    ]
    asset_img_filepath = f"neuraview/media/assets/{asset_icon_filename}"
    col12.image(asset_img_filepath, use_column_width=True)


def tariff_section_ui(agent_uid: str, agent_config: AgentConfig, locked: bool = False):
    col11, _, col12, col13 = st.columns((30, 4, 18, 18))
    current_tariff_str = agent_config.tariff
    available_tariff_list = AvailableTariffsEnum.list_tariffs()

    agent_config.tariff = col11.selectbox(
        "Select Tariff",
        available_tariff_list + ["General, ToU, Demand <50 kW"],
        key="tariff" + agent_uid,
        index=available_tariff_list.index(current_tariff_str),
        disabled=locked,
        label_visibility="collapsed",
    )

    agent_config.tariff = "Ontario TOU"

    tariff = AvailableTariffsEnum.from_string(agent_config.tariff)

    with col11:
        # st.write("**Description:** ", tariff.description)
        if tariff.kwh_range[0] is None or tariff.kwh_range[1] is None:
            energy_range = "None"
        else:
            energy_range = f"{tariff.kwh_range[0]} to {tariff.kwh_range[1]} kWh"
        if tariff.kw_range[0] is None or tariff.kw_range[1] is None:
            power_range = "None"
        else:
            power_range = f"{tariff.kw_range[0]} to {tariff.kw_range[1]} kW"

        tariff_df = pd.DataFrame(
            {
                "Class": [tariff.tariff_class],
                "Utility": [tariff.utility],
                "Billing Period": [tariff.charge_period],
                "Energy Range": [energy_range],
                "Power Range": [power_range],
            }
        ).T
        tariff_df["Characteristic"] = tariff_df.index
        tariff_df.columns = ["Value", "Characteristic"]
        st.dataframe(
            tariff_df[["Characteristic", "Value"]],
            use_container_width=True,
            hide_index=True,
        )

    with col12:
        st.write("##### Rate Criteria")
        st.checkbox(
            "Contracted",
            value=tariff.has_contracted_rates,
            disabled=True,
            key="contracted" + agent_uid,
        )
        st.checkbox(
            "Net Metering",
            value=tariff.has_net_metering,
            disabled=True,
            key="net_metering" + agent_uid,
        )
        st.checkbox(
            "Tiered",
            value=tariff.has_tiered_rates,
            disabled=True,
            key="tiered" + agent_uid,
        )
        st.checkbox(
            "Time of Use",
            value=tariff.has_time_of_use_rates,
            disabled=True,
            key="tou" + agent_uid,
        )
        st.checkbox(
            "Others",
            value=tariff.has_rate_applicability,
            disabled=True,
            key="others" + agent_uid,
        )

    with col13:
        st.write("##### Charge Types")
        st.checkbox(
            "Consumption",
            value=tariff.charge_type_consumption,
            disabled=True,
            key="consumption" + agent_uid,
        )
        st.checkbox(
            "Demand",
            value=tariff.charge_type_demand,
            disabled=True,
            key="demand" + agent_uid,
        )
        st.checkbox(
            "Fixed",
            value=tariff.charge_type_fixed,
            disabled=True,
            key="fixed" + agent_uid,
        )
        st.checkbox(
            "Minimum",
            value=tariff.charge_type_minimum,
            disabled=True,
            key="minimum" + agent_uid,
        )
        st.checkbox(
            "Maximum",
            value=tariff.charge_type_maximum,
            disabled=True,
            key="maximum" + agent_uid,
        )
        st.checkbox(
            "Quantity",
            value=tariff.charge_type_quantity,
            disabled=True,
            key="quantity" + agent_uid,
        )

    default_rate = tariff.base_rate
    if not tariff.has_time_of_use_rates:
        st.write("**Default Rate:**", default_rate, tariff.currency + "/kWh")
        rate_changes = []
    else:
        rate_changes = [
            {"start": 11, "end": 17, "days": [0, 1, 2, 3, 4], "value": 0.151},
            {"start": 7, "end": 11, "days": [0, 1, 2, 3, 4], "value": 0.102},
            {"start": 17, "end": 19, "days": [0, 1, 2, 3, 4], "value": 0.102},
        ]

    st.checkbox(
        "Show detailed charges", value=False, key="show_detailed_rate_info" + agent_uid
    )
    st.write("##### Time of Use")
    fig = create_tou_heatmap(default_rate, rate_changes=rate_changes)
    st.plotly_chart(fig, use_container_width=True)


def product_section_ui(agent_uid: str, agent_config: AgentConfig, locked: bool = False):
    col11, _, col12, _ = st.columns((9, 2, 8, 2))
    available_products_list = AvailableProductsEnum.list_products()
    if "prod_placeholder" in st.session_state:
        current_product_str = st.session_state["prod_placeholder"]
    else:
        current_product_str = agent_config.product

    PRODUCTS = {
        "Arbitrage": "Arbitrage",
        "Demand Response": "Demand Response",
        "Dynamic Pricing (CAISO)": "Arbitrage",
        "Tariff, GHG, and DR Optimization": "Decarbonization",
        "Energy Efficiency": "Energy Efficiency",
        "Grid Stability": "Grid Stability",
        "Load Flexibility": "Load Flexibility",
        "Power Peaks": "Power Peaks",
        "Tariff Optimization": "Tariff Optimization",
    }
    # REVERSE_PRODUCT = {v: k for k, v in PRODUCTS.items()}
    st.session_state["prod_placeholder"] = col11.selectbox(
        "Select Product",
        list(PRODUCTS.keys()) + ["HOEP Market Arbitrage"],
        list(PRODUCTS.keys()).index(current_product_str),
        key="product" + agent_uid,
        disabled=locked,
        label_visibility="collapsed",
    )
    st.session_state["prod_placeholder"] = "Arbitrage"
    sanitized_product = PRODUCTS[st.session_state["prod_placeholder"]]
    if sanitized_product in available_products_list:
        agent_config.product = sanitized_product

    with col11.popover(
        "See all available products",
    ):
        st.image("neuraview/media/products/all_products.png")

    with col11:
        col11.write("**Description:** " + PRODUCTS_DESCRIPTION_DICT[sanitized_product])
    col12.image(
        f"neuraview/media/products/{sanitized_product}.png", use_column_width=True
    )


def show_data_ui(agent_uid: str, agent_config: AgentConfig, locked: bool = False):
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "###### 🔧 Configuration",
            "###### 🧪 Sampling Synopsis",
            "###### 📉 Visualization",
            "###### 📂 Export",
        ],
    )

    # Signals Configuration
    with tab1:
        signals_df_dict = {
            "Data Signal": [s for s in agent_config.data.signals_info.keys()],
            "Tags": [s.tags for s in agent_config.data.signals_info.values()],
            "Min": [
                str(s.min_value) if s.min_value is not None else None
                for s in agent_config.data.signals_info.values()
            ],
            "Max": [
                str(s.max_value) if s.max_value is not None else None
                for s in agent_config.data.signals_info.values()
            ],
            "Time Range": [
                str((-float("inf"), 0)) for _ in agent_config.data.signals_info.values()
            ],
            "Scalable": [s.scalable for s in agent_config.data.signals_info.values()],
            "Source": [
                ("Product",) if "market" in s else ("Asset",)
                for s in agent_config.data.signals_info.keys()
            ],
        }

        signal_df = pd.DataFrame(signals_df_dict)
        import numpy as np

        signal_df.loc[1] = [
            "Outside Temperature",
            ["W", "S"],
            -50,
            50,
            (-float("inf"), 0),
            True,
            ("Provider",),
        ]

        signal_df.set_index("Data Signal", inplace=True)
        st.dataframe(signal_df, use_container_width=True)

    # Signal Sampling
    with tab2:
        if PRELOADED_AGENTS_KEY in st.session_state:
            df = st.session_state[PRELOADED_AGENTS_DF_KEY][agent_uid]
            st.dataframe(df.describe().T, use_container_width=True)
        else:
            st.warning("Please load the agent data to access this feature.")

    # Signal Visualization
    with tab3:
        if PRELOADED_AGENTS_KEY in st.session_state:
            st.write(" ")
            col11, col12 = st.columns(2, gap="large")

            # Get start date as simulation start
            start_datetime_str = st.session_state[SELECTED_SIM_SUMMARY_KEY][
                "time start"
            ]
            end_datetime_str = st.session_state[SELECTED_SIM_SUMMARY_KEY]["time end"]

            start_date = dt.datetime.strptime(start_datetime_str, DT_STR_FORMAT).date()
            end_date = dt.datetime.strptime(
                end_datetime_str, DT_STR_FORMAT
            ).date() + dt.timedelta(days=1)

            cols = col11.multiselect("Select Signals", list(df.columns))

            select_start, select_end = col12.slider(
                "Start Date",
                min_value=start_date,
                max_value=end_date,
                value=(start_date, end_date),
                key="start_date" + agent_uid,
            )

            show_plots_button = col11.button("Show Plots", key="show_plots" + agent_uid)

            if show_plots_button:
                st.write("#####")

                # Make start and end utc-aware
                # Convert date objects to datetime objects
                select_start_datetime = dt.datetime.combine(select_start, dt.time.min)
                select_end_datetime = dt.datetime.combine(select_end, dt.time.min)

                # Add timezone information
                select_start_utc = select_start_datetime.replace(tzinfo=dt.timezone.utc)
                select_end_utc = select_end_datetime.replace(tzinfo=dt.timezone.utc)

                sub_df = df.loc[select_start_utc:select_end_utc, cols]
                col21, col22 = st.columns(2, gap="medium")
                ts_fig = plotly_line_chart(sub_df, cols, show_legend=False)
                col21.plotly_chart(ts_fig)

                hist_fig = plotly_hist_plot(sub_df, cols, show_legend=False)
                col22.plotly_chart(hist_fig)


def show_agent_ui(agent_uid: str, agent_config: AgentConfig, locked: bool = False):
    # ASSET SECTION
    st.write("### 🔋 Asset Overview")
    asset_section_ui(agent_uid, agent_config, locked)
    st.divider()

    # TARIFF SECTION
    st.write("### 💸 Electricity Tariff")
    tariff_section_ui(agent_uid, agent_config, locked)
    st.divider()

    # PRODUCT SECTION
    st.write("### 🧳 Product Selection")
    product_section_ui(agent_uid, agent_config, locked)
    st.divider()

    # -----------------------------------------------------------------
    # DATA SECTION
    # -----------------------------------------------------------------
    st.write("### 🧵 Data and Metadata")
    show_data_ui(agent_uid, agent_config, locked)
    st.divider()

    st.write("### 🕹️ Control ")
    st.write("Under Development")

    # with st.form("agent_form_" + agent_uid, border=False):
    #     uid = col41.text_input(
    #         "Name (UID)", value=agent_uid, disabled=locked, key="uid" + agent_uid
    #     )
    #     asset_type = col42.selectbox(
    #         "Asset Type",
    #         AvailableAssetsEnum.list_assets_for_frontend(),
    #         key="asset_type" + agent_uid,
    #         disabled=locked,
    #     )

    #     with signal_popover:
    #         signal = st.selectbox(
    #             "Available Asset Signals",
    #             ["Internal Energy"],
    #             None,
    #             key="signal" + agent_uid,
    #             disabled=locked,
    #         )
    #         _, col51, col52, col53, col54, _ = st.columns((5, 10, 10, 10, 10, 2))
    #         X_tag = col51.checkbox("X", False, key="X" + agent_uid, disabled=locked)
    #         U_tag = col52.checkbox("U", False, key="U" + agent_uid, disabled=locked)
    #         W_tag = col53.checkbox("W", False, key="W" + agent_uid, disabled=locked)
    #         O_tag = col54.checkbox("O", False, key="O" + agent_uid, disabled=locked)

    #         active_signals = []
    #         if X_tag:
    #             active_signals.append("X")
    #         if U_tag:
    #             active_signals.append("U")
    #         if W_tag:
    #             active_signals.append("W")
    #         if O_tag:
    #             active_signals.append("O")

    #         col61, col62, _, col63 = st.columns(
    #             (12, 12, 4, 12), vertical_alignment="bottom"
    #         )

    #         min_value_known = col61.toggle(
    #             "Min Value Known", False, key="min_value_known" + agent_uid
    #         )
    #         min_value = col61.number_input(
    #             "Min Value",
    #             key="min_value" + agent_uid,
    #             disabled=locked or not min_value_known,
    #         )
    #         max_value_known = col62.toggle(
    #             "Max Value Known", False, key="max_value_known" + agent_uid
    #         )
    #         max_value = col62.number_input(
    #             "Max Value",
    #             key="max_value" + agent_uid,
    #             disabled=locked or not max_value_known,
    #         )
    #         scalable = col63.checkbox(
    #             "Scalable", key="scalable" + agent_uid, disabled=locked
    #         )

    #         if st.button("Add Signal", key="add_signal" + agent_uid, disabled=locked):
    #             agent_config.data.signals_info[signal] = SignalInfo(
    #                 tags=active_signals,
    #                 min_value=min_value,
    #                 max_value=max_value,
    #                 scalable=scalable,
    #             )

    #     if not locked:
    #         if st.form_submit_button("Add agent to simulation", type="primary"):
    #             st.session_state["new_simulation_config"].agents[uid] = agent_config
    #             st.rerun()
    # for agent in agents:
    #    with st.expander(f"**{agent.name}**"):
    #        col31, col32 = st.columns((1, 1))
    #        col31.write(f"Latitude: {agent.lat}")
    #        col32.write(f"Longitude: {agent.lon}")
    #        col33, col34 = st.columns((1, 1))
    #        col33.button("Edit")
    #        col34.button("Delete")
