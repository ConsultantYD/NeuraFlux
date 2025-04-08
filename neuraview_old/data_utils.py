import datetime as dt
import json
import os

import streamlit as st
from global_variables import (
    ALL_SIM_AGENTS_LIST_KEY,
    CONTROL_MODULE_KEY,
    DATA_MODULE_KEY,
    SELECTED_SIM_CONFIG_KEY,
    SELECTED_SIM_DIR_KEY,
    SELECTED_SIM_SUMMARY_KEY,
    SIM_SUMMARY_FILENAME,
    PRELOADED_AGENTS_KEY,
)
from neuraflux.agency.agent import Agent
from neuraflux.agency.control_module import ControlModule
from neuraflux.agency.data_module import DataModule
from neuraflux.global_variables import DT_STR_FORMAT
from neuraflux.schemas.agency import AgentConfig
from neuraflux.schemas.simulation import SimulationConfig
from neuraflux.time_ref import TimeInfo


def get_simulations_from_directory(directory: str = "simulations") -> list[str]:
    simulation_list = os.listdir(directory)
    sanitized_simulation_list = [
        s for s in simulation_list if os.path.isdir(os.path.join(directory, s))
    ]
    sanitized_simulation_list.sort()
    return sanitized_simulation_list


def load_simulation_config(simulation_dir: str) -> SimulationConfig:
    simulation_config_filepath = os.path.join(simulation_dir, "config.json")
    with open(simulation_config_filepath) as f:
        simulation_config_dict = json.load(f)
    return SimulationConfig.model_validate(simulation_config_dict)


def load_config_file(config_filepath: str) -> SimulationConfig:
    with open(config_filepath) as f:
        simulation_config_dict = json.load(f)
    return SimulationConfig.model_validate(simulation_config_dict)


def load_simulation_summary(simulation_dir: str) -> dict[str, str]:
    simulation_summary_filepath = os.path.join(simulation_dir, SIM_SUMMARY_FILENAME)
    with open(simulation_summary_filepath) as f:
        simulation_summary_dict = json.load(f)
    return simulation_summary_dict


def get_control_module(sim_name: str) -> ControlModule:
    control_module = ControlModule(base_dir=sim_name)
    return control_module


def get_data_module(sim_name: str) -> DataModule:
    data_module = DataModule(base_dir=sim_name)
    return data_module


def load_data_in_session(selected_sim_dir: str) -> None:
    new_sim: bool = False

    # Selected simulation is not in memory
    if (
        SELECTED_SIM_DIR_KEY not in st.session_state
        or st.session_state[SELECTED_SIM_DIR_KEY] != selected_sim_dir
    ):
        new_sim = True

        # Update all simulation-related session state variables
        st.session_state[SELECTED_SIM_DIR_KEY] = selected_sim_dir
        st.session_state[SELECTED_SIM_CONFIG_KEY] = load_simulation_config(
            selected_sim_dir
        )

        # Update list of all agents in the simulation for convenience
        st.session_state[ALL_SIM_AGENTS_LIST_KEY] = list(
            st.session_state[SELECTED_SIM_CONFIG_KEY].agents.keys()
        )

    # Always update the simulation summary (can change every second)
    st.session_state[SELECTED_SIM_SUMMARY_KEY] = load_simulation_summary(
        selected_sim_dir
    )

    # Data module
    if DATA_MODULE_KEY not in st.session_state or new_sim:
        data_module = get_data_module(selected_sim_dir)
        st.session_state["data_module"] = data_module

    # Control module
    if CONTROL_MODULE_KEY not in st.session_state or new_sim:
        control_module = get_control_module(selected_sim_dir)
        st.session_state["control_module"] = control_module


def get_agent_in_active_simulation(agent_uid: str) -> Agent:
    # Agent Config
    sim_dir = st.session_state[SELECTED_SIM_DIR_KEY]
    agent_config_filepath = os.path.join(sim_dir, agent_uid, "config.json")
    with open(agent_config_filepath) as f:
        agent_config_json = json.load(f)
        agent_config = AgentConfig(**agent_config_json)

    # Modules
    control_module = st.session_state[CONTROL_MODULE_KEY]
    data_module = st.session_state[DATA_MODULE_KEY]

    # Time Info
    sim_summary = st.session_state[SELECTED_SIM_SUMMARY_KEY]
    # sim_start = dt.datetime.strptime(sim_summary["time start"], DT_STR_FORMAT)
    # sim_end = dt.datetime.strptime(sim_summary["time end"], DT_STR_FORMAT)
    current_time = dt.datetime.strptime(sim_summary["current time"], DT_STR_FORMAT)
    time_info = TimeInfo(current_time, dt.timedelta(minutes=5))

    # Initialize agent
    agent = Agent(agent_uid, agent_config, data_module, control_module, time_info, None)
    return agent


def clear_simulation_data_from_session_state():
    for key in [
        SELECTED_SIM_DIR_KEY,
        SELECTED_SIM_CONFIG_KEY,
        SELECTED_SIM_SUMMARY_KEY,
        ALL_SIM_AGENTS_LIST_KEY,
        CONTROL_MODULE_KEY,
        DATA_MODULE_KEY,
    ]:
        if key in st.session_state:
            del st.session_state[key]
