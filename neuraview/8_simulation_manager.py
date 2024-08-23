import datetime as dt

import pandas as pd
import streamlit as st
from data_utils import get_agent_in_active_simulation
from global_variables import SELECTED_SIM_CONFIG_KEY, SELECTED_SIM_SUMMARY_KEY
from neuraflux.global_variables import DT_STR_FORMAT, PRICE_KEY
from sidebar_utils import generate_sidebar

# ---------------------------------------------------------------------------
# INITIALIZATION SECTION
# ---------------------------------------------------------------------------
# Page title and sidebar initialization
st.title("🏔️ Simulation Manager")
generate_sidebar()

# Retrieve simulation summary information
simulation_summary = st.session_state[SELECTED_SIM_SUMMARY_KEY]
sim_start = dt.datetime.strptime(simulation_summary["time start"], DT_STR_FORMAT)
sim_end = dt.datetime.strptime(simulation_summary["time end"], DT_STR_FORMAT)
sim_t = dt.datetime.strptime(simulation_summary["current time"], DT_STR_FORMAT)

# ---------------------------------------------------------------------------
# SIMULATION PROGRESS SECTION
# ---------------------------------------------------------------------------
st.write("### Simulation Progress")
st.write("###### Current time: ", sim_t)
col11, col12, col13 = st.columns([4, 17, 4])
col11.write("###### " + sim_start.strftime("%Y-%m-%d"))
col13.write("###### " + sim_end.strftime("%Y-%m-%d"))
global_delta_s = (sim_end - sim_start).total_seconds()
current_delta_s = (sim_t - sim_start).total_seconds()

# Progress bar
_, col22, _ = st.columns([1.5, 17, 2])
col22.progress(current_delta_s / global_delta_s)

st.divider()


# ---------------------------------------------------------------------------
# PERFORMANCE OVERVIEW SECTION
# ---------------------------------------------------------------------------
st.write("### Performance Overview")

with st.spinner("Loading performance data..."):
    agents_uid_list = list(st.session_state[SELECTED_SIM_CONFIG_KEY].agents.keys())
    # rewards = {}
    dataframes = {}
    for agent_uid in agents_uid_list:
        agent = get_agent_in_active_simulation(agent_uid)
        dataframes[agent_uid] = agent.get_data(end_time=sim_t)
        # rewards[agent_uid] = agent.get_reward_data(end_time=sim_t)

    all_prices = [df[PRICE_KEY].values for df in dataframes.values()]
    averaged_prices = [
        df[PRICE_KEY].rolling(12 * 24).mean().dropna().values
        for df in dataframes.values()
    ]

    # reward_arrays = [rewards[agent_uid]["reward"].values for agent_uid in agents_uid_list]
    # averaged_reward_arrays = [
    #            rewards[agent_uid]["reward"].rolling(12*24).mean().dropna().values for agent_uid in agents_uid_list
    #        ]

    df_summary = pd.DataFrame(
        {
            "Agent": agents_uid_list,
            "Status": ["Active"] * len(agents_uid_list),
            "Recent Rewards": averaged_prices,
            "Revenue": [prices.sum() * -1 for prices in all_prices],
        }
    )
    df_summary.set_index("Agent", inplace=True)
    st.dataframe(
        df_summary,
        column_config={
            "Recent Rewards": st.column_config.AreaChartColumn(
                "Cumulative Profit (past 30 days)", y_min=0, y_max=1, width="large"
            ),
            "Revenue": st.column_config.NumberColumn(
                "Revenue", format="%.2f 💰", width="small"
            ),
        },
        use_container_width=True,
    )
st.divider()

# Agents
st.write("### Agents")
