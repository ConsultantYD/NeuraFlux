import os

import streamlit as st
from data_utils import (
    get_simulations_from_directory,
    load_data_in_session,
    get_agent_in_active_simulation,
)
from global_variables import (
    SELECTED_SIM_NAME_KEY,
    SIMS_ROOT_DIR,
    PRELOADED_AGENTS_LIST_KEY,
    PRELOADED_AGENTS_KEY,
    PRELOADED_AGENTS_DF_KEY,
    ALL_SIM_AGENTS_LIST_KEY,
)


def generate_sidebar():
    st.sidebar.title("NeuraView Dashboard")
    st.sidebar.write("## Simulation")
    col01, col02 = st.sidebar.columns(2)

    if "new_sim_clicked" not in st.session_state:
        st.session_state["new_sim_clicked"] = False
    if "load_sim_clicked" not in st.session_state:
        st.session_state["load_sim_clicked"] = False

    new_sim_clicked = col01.button(
        "❇️ New Sim",
        use_container_width=True,
        disabled=st.session_state["new_sim_clicked"],
    )
    load_sim_clicked = col02.button(
        "📥 Load Sim",
        use_container_width=True,
        disabled=st.session_state["load_sim_clicked"],
    )

    if new_sim_clicked:
        st.session_state["new_sim_clicked"] = True
        st.session_state["load_sim_clicked"] = False
        st.switch_page("9_simulation_config.py")
    if load_sim_clicked:
        st.session_state["new_sim_clicked"] = False
        st.session_state["load_sim_clicked"] = True
        st.rerun()

    if st.session_state["load_sim_clicked"]:
        # Existing simulation selection
        available_simulations_list = get_simulations_from_directory(SIMS_ROOT_DIR)
        sim_idx = None
        if SELECTED_SIM_NAME_KEY in st.session_state:
            sim_idx = available_simulations_list.index(
                st.session_state[SELECTED_SIM_NAME_KEY]
            )
        selected_sim_name = st.sidebar.selectbox(
            "Load existing simulation",
            available_simulations_list,
            index=sim_idx,
        )
        if selected_sim_name is not None:
            st.session_state[SELECTED_SIM_NAME_KEY] = selected_sim_name

            # Load simulation data
            selected_sim_full_dir = os.path.join(SIMS_ROOT_DIR, selected_sim_name)
            with st.spinner("Loading ..."):
                load_data_in_session(selected_sim_full_dir)

            # PRE-LOAD AGENTS DATA
            # Get list of all agents in simulation
            all_agents_list = st.session_state[ALL_SIM_AGENTS_LIST_KEY]
            st.session_state[PRELOADED_AGENTS_LIST_KEY] = st.sidebar.multiselect(
                "Load agents data",
                all_agents_list,
                st.session_state.get(PRELOADED_AGENTS_LIST_KEY, []),
            )

            col11, col12 = st.sidebar.columns((5, 6))
            load_agents_button = col11.button("Load agents")
            refresh_agent_data = col12.toggle("Refresh data", value=False)

            if load_agents_button:
                # Make sure the preloaded agents dict exists
                if PRELOADED_AGENTS_KEY not in st.session_state:
                    st.session_state[PRELOADED_AGENTS_KEY] = {}

                if PRELOADED_AGENTS_DF_KEY not in st.session_state:
                    st.session_state[PRELOADED_AGENTS_DF_KEY] = {}

                # Loop over selected agents and pre-load their data
                for agent_uid in st.session_state[PRELOADED_AGENTS_LIST_KEY]:
                    # If agent not in memory, load it
                    if (
                        agent_uid not in st.session_state[PRELOADED_AGENTS_KEY]
                        or refresh_agent_data
                    ):
                        with st.sidebar:
                            with st.spinner(f"Loading {agent_uid} data ..."):
                                agent = get_agent_in_active_simulation(agent_uid)
                                st.session_state[PRELOADED_AGENTS_KEY][agent_uid] = (
                                    agent
                                )

                                df = agent.get_data(
                                    tariff_data=True,
                                    product_data=True,
                                    time_features=True,
                                )
                                st.session_state[PRELOADED_AGENTS_DF_KEY][agent_uid] = (
                                    df
                                )
    st.sidebar.info("Developed by Ysael Desage.")
