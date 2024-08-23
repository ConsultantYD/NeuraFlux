import streamlit as st
from global_variables import PRELOADED_AGENTS_DF_KEY, PRELOADED_AGENTS_KEY
from plot_utils import (
    create_radar_plot,
    plotly_colored_line_chart,
    plotly_filled_grad_line_chart,
    plotly_sankey_plot,
)
from sidebar_utils import generate_sidebar

# Page title and sidebar initialization
st.header("🚀 Performance & Profitability")
generate_sidebar()


agents_uid_list = st.session_state[PRELOADED_AGENTS_KEY]
col001, _ = st.columns((3, 4))
agent_uid = col001.selectbox("Select Agent", agents_uid_list, None)

st.divider()

if agent_uid is not None:
    df = st.session_state[PRELOADED_AGENTS_DF_KEY][agent_uid]

    green_colorscale = [
        [0, "rgba(255, 255, 255, 0.1)"],  # White
        [1, "rgba(11, 145, 0, 0.5)"],  # Green
    ]
    reversed_green_colorscale = green_colorscale = [
        [0, "rgba(11, 145, 0, 0.5)"],  # Green
        [1, "rgba(255, 255, 255, 0.1)"],  # White
    ]

    blue_colorscale = [
        [0, "rgba(255, 255, 255, 0.1)"],  # White
        [1, "rgba(0, 0, 255, 0.5)"],  # Blue
    ]

    gold_colorscale = [
        [0, "rgba(255, 255, 255, 0.1)"],  # White
        [1, "rgba(255, 215, 0, 0.5)"],  # Gold
    ]

    df["cum_reward"] = df["reward"].cumsum()
    df["avg_reward"] = df["reward"].rolling(window=12 * 24).mean()
    df["cum_price"] = df["price_$"].cumsum()
    df["avg_price"] = df["price_$"].rolling(window=12 * 24).mean()
    df["cum_profit"] = -df["cum_price"].values / 750
    df["avg_profit"] = df["avg_price"].values / 1.5

    # -----------------------------------------------------------------
    # PROFITABILITY
    # -----------------------------------------------------------------
    st.write("##### Profitability")
    st.warning(
        "**Tangible value** gained from deploying the **NeuraFlux Agent** compared to the baseline, encompassing benefits from increased **efficiency**, **cost savings**, and new **financial opportunities**."
    )

    col11, _, col12, _, col13, _ = st.columns(
        (14, 2, 4, 1, 4, 1), vertical_alignment="center"
    )

    # Prepare figures and data
    fig1 = plotly_colored_line_chart(
        df,
        "avg_profit",
        show_legend=False,
        height=200,
        line_color="gold",
    )
    fig2 = plotly_filled_grad_line_chart(
        df,
        "cum_profit",
        gold_colorscale,
        line_color="gold",
        show_legend=False,
        height=200,
    )

    global_daily_average_profit = df["avg_profit"].mean()
    last_daily_average_profit = df["avg_profit"].iloc[-288:].mean()
    last_weekly_average_profit = df["avg_profit"].iloc[-2016:].mean()

    # Display Metrics
    col12.metric(
        "Daily Average",
        value="10.3$",
        delta="+2.09",
    )
    col12.write("#####")
    col13.metric(
        "Weekly Average",
        value="68.9$",
        delta=-0.32,
    )
    col13.write("#####")

    # Display Figures
    with col11:
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "**Cumulative**",
                "**Net Margin**",
                "**Breakdown**",
                "**Risk & Volatility**",
            ]
        )
        with tab1:
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------------------------------------------
    # FINANCIAL FLOW
    # -----------------------------------------------------------------
    st.write("##### Financial Flow")
    st.success(
        "Analysis of the **NeuraFlux Agent**'s deployment on the stakeholder's **financial dynamics**, including **expenses**, **income**, and **liquidity** impacts."
    )

    col21, _, col22, _, col23, _ = st.columns(
        (24, 4, 8, 1, 8, 1), vertical_alignment="center"
    )

    # Prepare figures and data
    fig1 = plotly_colored_line_chart(
        df,
        "avg_price",
        show_legend=False,
        height=200,
    )
    fig2 = plotly_filled_grad_line_chart(
        df,
        "cum_price",
        reversed_green_colorscale,
        line_color="green",
        show_legend=False,
        height=200,
    )
    sankey_fig = plotly_sankey_plot()

    global_daily_average_profit = df["avg_price"].mean()
    last_daily_average_profit = df["avg_price"].iloc[-288:].mean()
    last_weekly_average_profit = df["avg_price"].iloc[-2016:].mean()

    # Display Metrics
    col22.metric(
        "Current Period",
        value="-963$",
        delta="+12.3",
    )
    col22.write("#####")
    col23.metric(
        "Monthly Avg.",
        value="-1262$",
        delta="+18.35",
    )
    col23.write("#####")

    # Display Figures
    with col21:
        tab21, tab22, tab23 = st.tabs(
            ["**Cash Flow**", "**Revenue Streams**", "**Expenses Allocation**"]
        )
        with tab21:
            st.plotly_chart(fig2, use_container_width=True)

        with tab22:
            st.plotly_chart(fig1, use_container_width=True)

        with tab23:
            st.plotly_chart(sankey_fig, use_container_width=True)

    # -----------------------------------------------------------------
    # REWARD PLOTS
    # -----------------------------------------------------------------
    st.write("##### Reward Signals")
    st.info(
        "**Reinforcement signals** generated and **maximized** by the **NeuraFlux Agent**, focusing on their **alignment** with stakeholder **objectives** and their **accrual** over different time spans."
    )

    col31, _, col32, _, col33, _ = st.columns(
        (14, 2, 4, 1, 4, 1), vertical_alignment="center"
    )

    fig3 = plotly_colored_line_chart(
        df, "avg_reward", line_color="blue", show_legend=False, height=200
    )
    fig4 = plotly_filled_grad_line_chart(
        df,
        "cum_reward",
        blue_colorscale,
        line_color="blue",
        show_legend=False,
        height=200,
    )

    # Sample dataset
    baseline = {
        "Comfort": 2,
        "Power<br>Peaks": 4,
        "Energy<br>Efficiency": 3,
        "Emissions<br>Reduction": 1,
        "Equipment<br>Health": 5,
        "Financial<br>Expenses": 2,
    }

    max_values = {
        "Comfort": 10,
        "Power<br>Peaks": 10,
        "Energy<br>Efficiency": 10,
        "Emissions<br>Reduction": 10,
        "Equipment<br>Health": 10,
        "Financial<br>Expenses": 10,
    }

    current = {
        "Comfort": 7,
        "Power<br>Peaks": 9,
        "Energy<br>Efficiency": 8,
        "Emissions<br>Reduction": 7,
        "Equipment<br>Health": 9.5,
        "Financial<br>Expenses": 7,
    }

    with col31:
        tab31, tab32, tab33 = st.tabs(
            ["**Multi-Objectives**", "**Empirical Distribution**", "**Growth Curves**"]
        )

    col32.metric(
        "Financial Expenses",
        value="7.8",
        delta="+1.09",
    )
    col33.metric(
        "Emissions Reduction",
        value="8.3",
        delta=-0.12,
    )
    col32.metric(
        "Energy Efficiency",
        value="8.5",
        delta="+1.01",
    )
    col33.metric(
        "Power Peaks",
        value="9.1",
        delta=0.00,
    )
    col32.metric(
        "Comfort",
        value="6.9",
        delta="-0.52",
    )
    col33.metric(
        "Equipment Health",
        value="8.9",
        delta=-0.32,
    )

    fig = create_radar_plot(baseline, max_values, current)
    tab31.write(fig)
