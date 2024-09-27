import datetime as dt

import streamlit as st
from global_variables import DATA_MODULE_KEY, SELECTED_SIM_CONFIG_KEY
from sidebar_utils import generate_sidebar
from agent_utils import show_agent_ui
from data_utils import load_config_file

# Page title and sidebar initialization
st.header("♟️ Agent Profile")
generate_sidebar()

if DATA_MODULE_KEY in st.session_state:
    # Variables initialization
    with st.spinner("Loading performance data..."):
        agents_uid_list = list(st.session_state[SELECTED_SIM_CONFIG_KEY].agents.keys())

    col00, _ = st.columns((3, 4), vertical_alignment="bottom")
    agent_uid = col00.selectbox("Select Agent", agents_uid_list, None)

    st.divider()
    if agent_uid is not None:
        show_agent_ui(
            agent_uid,
            st.session_state[SELECTED_SIM_CONFIG_KEY].agents[agent_uid],
            locked=True,
        )

    if agent_uid is not None:
        col11, col12, col13 = st.columns((1, 1, 2))
        start_date = col11.date_input("Start Date", value=dt.date(2023, 1, 1))
        end_date = col12.date_input("End Date", value=dt.date(2023, 1, 7))
        # Convert date to datetime
        start_datetime = dt.datetime.combine(start_date, dt.time(0, 0, 0))
        end_datetime = dt.datetime.combine(
            end_date - dt.timedelta(days=1), dt.time(23, 59, 59)
        )

else:
    config = load_config_file("config.json")
    show_agent_ui(
        agent_uid="",
        agent_config=config.agents["Agent001"],
        locked=False,
    )
