import datetime as dt
import os

import streamlit as st
from streamlit import session_state as ss
from global_variables import (
    SELECTED_SIM_NAME_KEY,
    SIMS_ROOT_DIR,
)

from neuraflux.schemas.agency import AgentConfig
from neuraflux.agency.agent import Agent


def generate_sidebar():
    st.sidebar.title("NeuraView Dashboard")

    # --------------------------------------------------------
    # - SIMULATION SELECTION
    # --------------------------------------------------------
    st.sidebar.write("## Simulation")
    ss[SELECTED_SIM_NAME_KEY] = st.sidebar.selectbox(
        "Select Simulation",
        options=[" "]
        + [
            sim
            for sim in os.listdir(SIMS_ROOT_DIR)
            if os.path.isdir(os.path.join(SIMS_ROOT_DIR, sim))
        ],
        key="sim_selection",
        label_visibility="collapsed",
    )
    if ss[SELECTED_SIM_NAME_KEY] == " ":
        st.sidebar.file_uploader(
            "Upload Simulation",
            type=["zip"],
            label_visibility="collapsed",
            key="sim_file_upload",
        )

    else:
        # --------------------------------------------------------
        # - AGENT(S) SELECTION
        # --------------------------------------------------------
        ss.sim_dir = os.path.join(
            SIMS_ROOT_DIR, st.session_state[SELECTED_SIM_NAME_KEY]
        )
        st.sidebar.write("## Agents")
        sim_folder_content = os.listdir(
            os.path.join(SIMS_ROOT_DIR, st.session_state[SELECTED_SIM_NAME_KEY])
        )
        available_agents = [
            agent
            for agent in sim_folder_content
            if os.path.isdir(os.path.join(ss.sim_dir, agent))
        ]
        ss.selected_agent = st.sidebar.multiselect(
            "Select Agent",
            options=available_agents,
            label_visibility="collapsed",
        )

        if ss.selected_agent:
            agent_dir = os.path.join(ss.sim_dir, ss.selected_agent[0])
            ss.agent = Agent.from_dir(agent_dir)

        st.sidebar.info("Developed by Ysael Desage.")
