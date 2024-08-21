import joypy
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_utils import get_agent_in_active_simulation
from global_variables import SELECTED_SIM_CONFIG_KEY
from sidebar_utils import generate_sidebar


def plot_td_rmse(
    df: pd.DataFrame, show_target_updates: bool = False, show_model_changes: bool = True
) -> go.Figure:
    # Create the main time series plot for the "td_rmse" column
    fig = go.Figure(
        go.Scatter(x=df.index, y=df["td_rmse"], mode="lines", name="TD RMSE")
    )

    # Set the background to white
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=0, b=60),
    )

    # TARGET ITERATOR LINES
    if show_target_updates:
        # Identify the indices where "target_iterator" changes
        change_points = df["target_iterator"].diff() != 0

        # Add vertical lines at these change points
        for change_point in df.index[change_points]:
            fig.add_vline(x=change_point, line=dict(color="lightgrey"))

    # MODEL CHANGE LINES
    if show_model_changes:
        # Identify changes in the string column
        string_changes = df["model_name"] != df["model_name"].shift()
        # Add vertical lines at these change points for the string column
        for change_point in df.index[string_changes]:
            fig.add_vline(x=change_point, line=dict(color="red"))

    return fig


def plot_reward_time_series(
    df: pd.DataFrame,
    reward_col: str,
    training_timestamps: list[int],
    show_training_times: bool = True,
    show_percentile_band: bool = True,
    lower_percentile=0.25,
    upper_percentile=0.75,
) -> go.Figure:
    # Calculate the 24-hour rolling mean
    rolling_mean = df[reward_col].rolling("24H").mean()

    # Create the main time series plot with the rolling mean
    fig = go.Figure(
        go.Scatter(
            x=df.index,
            y=rolling_mean,
            mode="lines",
            name="24H Rolling Mean",
            line=dict(color="blue"),
        )
    )

    # Add shaded area for the percentiles if enabled
    if show_percentile_band:
        # Calculate the percentiles within the rolling window
        rolling_lower = df[reward_col].rolling("24H").quantile(lower_percentile)
        rolling_upper = df[reward_col].rolling("24H").quantile(upper_percentile)

        # Convert indices and data to Series for concatenation
        x_series = pd.Series(df.index, index=df.index)
        y_upper = pd.Series(rolling_upper, index=df.index)
        y_lower = pd.Series(rolling_lower, index=df.index)

        fig.add_trace(
            go.Scatter(
                x=pd.concat([x_series, x_series[::-1]]),  # x, then x reversed
                y=pd.concat([y_upper, y_lower[::-1]]),  # upper, then lower reversed
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{int(lower_percentile*100)}-{int(upper_percentile*100)} Percentile Band",
            )
        )

    # Plot vertical lines for training timestamps
    if show_training_times:
        for ts in training_timestamps:
            # Convert UNIX timestamp to datetime
            time_point = pd.to_datetime(ts, unit="s", utc=True)
            fig.add_vline(
                x=time_point,
                line=dict(color="red"),
                name="Training Timestamp",
            )

    # Set the background to white and adjust layout
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=0, b=60),
        title="Reward Time Series with 24H Rolling Mean and Training Timestamps",
    )

    return fig


def duration_joyplot(column: str, group_by: str = "model_name"):
    fig, ax = joypy.joyplot(
        training_table,
        by=group_by,
        column=column,
        range_style="own",
        grid="y",
        linewidth=1,
        legend=False,
        fade=True,
    )
    return fig


# Page title and sidebar initialization
st.header("🧬 Learning & Evolution")
generate_sidebar()

if SELECTED_SIM_CONFIG_KEY in st.session_state:
    # Variables initialization
    with st.spinner("Loading performance data..."):
        agents_uid_list = list(st.session_state[SELECTED_SIM_CONFIG_KEY].agents.keys())

    col00, col01 = st.columns(2)
    agent_uid = col00.selectbox("Select Agent", agents_uid_list, None)

    if agent_uid is not None:
        agent = get_agent_in_active_simulation(agent_uid)
        training_summary = agent.get_training_summary()
        training_table = agent.get_rl_training_data_table()

        # ------------------- #
        # REWARD
        # ------------------- #
        st.subheader("💰 Reward")
        col11, col12, col13 = st.columns(3, vertical_alignment="center")
        show_training_times = col11.toggle("Show Training Times", True)
        show_percentile_band = col11.toggle("Show Percentile Band", False)

        lower_percentile = 0.25
        upper_percentile = 0.75
        if show_percentile_band:
            lower_percentile = col12.slider("Lower Percentile", 0.0, 0.49, 0.25, 0.01)
            upper_percentile = col13.slider("Upper Percentile", 0.51, 1.0, 0.75, 0.01)
        reward_df = agent.get_reward_data()
        training_timestamps = training_table["training_timestamp"].unique().tolist()
        reward_fig = plot_reward_time_series(
            reward_df,
            "reward",
            training_timestamps,
            show_training_times,
            show_percentile_band,
            lower_percentile,
            upper_percentile,
        )
        st.plotly_chart(reward_fig, use_container_width=True)
        st.divider()

        # ------------------- #
        # TD ERROR
        # ------------------- #
        st.subheader("🎯 TD Error")

        # Define data subset based on wanted models
        all_models = sorted(training_table["model_name"].unique())
        models_selected = st.multiselect(
            "Select Models to Show",
            all_models,
            all_models,
        )
        col21, col22, _ = st.columns(3)
        show_model_changes = col21.toggle("Show Model Changes", True)
        show_target_updates = col22.toggle("Show Target Updates", False)
        sub_df = training_table[training_table["model_name"].isin(models_selected)]

        # Plot
        if st.button("Show TD Error Plot"):
            training_ts_plot = plot_td_rmse(
                sub_df, show_target_updates, show_model_changes
            )
            st.plotly_chart(training_ts_plot, use_container_width=True)
        st.divider()

        # ------------------- #
        # EXECUTION TIME
        # ------------------- #
        st.subheader("⏳ Execution Time")
        col31, col32 = st.columns(2)
        fit_duration_fig = duration_joyplot("training_duration")
        batch_sampling_duration_fig = duration_joyplot("batch_sampling_duration")
        global_td_errors_calculations_duration_fig = duration_joyplot(
            "global_td_errors_calculation_duration"
        )
        td_error_PER_update_duration_fig = duration_joyplot(
            "td_error_PER_update_duration"
        )

        col31.write("##### TensorFlow Fit Duration")
        col31.pyplot(fit_duration_fig)
        col32.write("##### Exp Batch Sampling Duration")
        col32.pyplot(batch_sampling_duration_fig)
        col41, col42 = st.columns(2)
        col41.write("##### TD Error Calculations Duration")
        col41.pyplot(global_td_errors_calculations_duration_fig)
        col42.write("##### TD Error PER Update Duration")
        col42.pyplot(td_error_PER_update_duration_fig)


else:
    st.warning("No data loaded. Please select a simulation.")
