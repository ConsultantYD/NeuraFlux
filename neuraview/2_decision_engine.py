import streamlit as st
from sidebar_utils import generate_sidebar
from global_variables import (
    PRELOADED_AGENTS_KEY,
    PRELOADED_AGENTS_DF_KEY,
)
from neuraflux.agency.agent import Agent
from plot_utils import (
    plot_q_factors_as_bars,
    plot_q_factors_as_circles,
    create_dual_violin_plot,
    create_scenario_tree_plot,
    create_return_hist_plot,
)
import pandas as pd
import numpy as np

# Page title and sidebar initialization
st.header("🧠 Decision Engine")
generate_sidebar()

agents_uid_list = st.session_state[PRELOADED_AGENTS_KEY]
col001, _ = st.columns((3, 4))
agent_uid = col001.selectbox("Select Agent", agents_uid_list, None)

st.divider()

if agent_uid is not None:
    agent: Agent = st.session_state[PRELOADED_AGENTS_KEY][agent_uid]
    df = st.session_state[PRELOADED_AGENTS_DF_KEY][agent_uid]

    rl_df = df.copy()

    # ----------------------------------------------------------------
    # EXPERIENCE DENSITY PLOT
    # ----------------------------------------------------------------
    st.write("#### Experience Density")
    st.write(
        "**Real experience** is shown in :orange[**orange**], **simulated experience** in :blue[**blue**]. The :red[**red**] dot marks the **current state value**, indicating the agent’s experience and the potential reliability of the estimate in this domain."
    )
    col21, col22, col23, col24 = st.columns(4)
    columns = [
        "outside_air_temperature",
        "internal_energy",
        "market_price_t",
        "tf_cos_m",
    ]
    ui_columns = [
        "<b>Temperature",
        "<b>Internal Energy",
        "<b>Market Price",
        "<b>Seasonality",
    ]
    for i in range(4):
        # Create an artificial dataset sampling between min and max values of col
        if i % 4 == 0:
            # Normal Distribution
            fake_data = np.random.normal(
                loc=df[columns[i]].mean(), scale=df[columns[i]].std(), size=len(df)
            )
        elif i % 4 == 1:
            # Exponential Distribution
            fake_data = np.random.uniform(
                df[columns[i]].min(), df[columns[i]].max(), size=len(df)
            )
        elif i % 4 == 2:
            # Log-Normal Distribution
            fake_data = np.random.lognormal(
                mean=np.log(df[columns[i]].mean()),
                sigma=df[columns[i]].std() * 3,
                size=len(df),
            )
        elif i % 4 == 3:
            # Beta Distribution
            fake_data = (
                np.random.beta(a=2, b=5, size=len(df))
                * (df[columns[i]].max() - df[columns[i]].min())
                + df[columns[i]].min()
            )

        df[ui_columns[i]] = fake_data
        fig = create_dual_violin_plot(
            df,
            ui_columns[i],
            columns[i],
            df.iloc[-1:, :],
            column_ui=ui_columns[i],
            height=200,
        )
        if i % 4 == 0:
            col21.plotly_chart(fig, use_container_width=True)
        elif i % 4 == 1:
            col22.plotly_chart(fig, use_container_width=True)
        elif i % 4 == 2:
            col23.plotly_chart(fig, use_container_width=True)
        else:
            col24.plotly_chart(fig, use_container_width=True)
    st.write("####")
    # ----------------------------------------------------------------
    # AGENT BRAIN EXPLORER
    # ----------------------------------------------------------------
    st.write("#### Scenario Playground")
    st.write(
        "Explore different scenarios by modifying the state on the left and see how the agent predicts long-term outcomes, assessing the risk and potential gains for each action along with possible outcomes."
    )

    col311, col312, col313 = st.columns((1, 1, 1))
    col311.selectbox("Select **Strategy** (Policy)", ["Q-Factors Argmax"], 0)
    col312.selectbox("Select **Q-Network**", ["Transformer 02-15"], 0)
    col313.selectbox("Select **Simulation Model**", ["Physics-Based"], 0)

    st.write("######")
    col31, col32 = st.columns((1, 2), gap="large")

    col31.write("##### State Definition")
    col31.toggle("Simplified View", value=True)
    col31.date_input("Date", value=df.index[-1])
    col31.time_input("Time", value=df.index[-1])
    col31.number_input(
        "Internal Energy (kWh)", value=int(df["internal_energy"].iloc[-1] + 150)
    )
    col31.number_input(
        "Outside Air Temperature (ºC)",
        value=df["outside_air_temperature"].iloc[-1] + 15,
    )
    col31.number_input("Market Price ($/MWh)", value=7.93, step=0.01, format="%.2f")

    col32.write("##### Action Potential")
    q_factors = agent.get_q_factors(rl_df=rl_df.iloc[0:10])
    q_factors_sanitized = q_factors[0][0].ravel()
    q_factors_sanitized = [100, 90, -40]
    fig = plot_q_factors_as_circles(q_factors_sanitized)
    col32.plotly_chart(fig, use_container_width=True)

    col32.write("##### Return Probabilities")
    fig = create_return_hist_plot()
    col32.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------
    # SCENARIO TREE EXPANSION
    # ----------------------------------------------------------------
    st.write("##### Scenario Tree Expansion")
    fig = create_scenario_tree_plot()
    st.plotly_chart(fig)

    col11, col12, col13 = st.columns(3)
    figs = []
    for i in range(len(q_factors[0])):
        figs.append(plot_q_factors_as_bars(q_factors[0][i].ravel()))

        # Plot in the right column
        if i % 3 == 0:
            col11.plotly_chart(figs[i])
        elif i % 3 == 1:
            col12.plotly_chart(figs[i])
        else:
            col13.plotly_chart(figs[i])
else:
    st.warning("No data loaded. Please select a simulation.")
