import streamlit as st
from global_variables import (
    PRELOADED_AGENTS_DF_KEY,
    PRELOADED_AGENTS_KEY,
    PRELOADED_SHADOW_ASSET_DF_KEY,
)
from plot_utils import (
    create_radar_plot,
    plotly_colored_line_chart,
    plotly_filled_grad_line_chart,
    plotly_sankey_plot,
    create_profit_hist_plot,
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
    shadow_df = st.session_state[PRELOADED_SHADOW_ASSET_DF_KEY][agent_uid]

    green_colorscale = [
        [0, "rgba(255, 255, 255, 0.1)"],  # White
        [1, "rgba(11, 145, 0, 0.5)"],  # Green
    ]
    reversed_green_colorscale = [
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
    df["cum_cash_flow"] = -df["price_$"].cumsum()
    df["avg_cash_flow"] = -df["price_$"].rolling(window=12 * 24).mean()
    df["hourly_cash_flow"] = -df["price_$"].rolling(window=12).sum()

    df["profit"] = shadow_df["price_$"] - df["price_$"]
    df["hourly_profit"] = df["profit"].rolling(window=12).sum()
    df["daily_profit"] = df["profit"].rolling(window=12 * 24).sum()
    df["cum_profit"] = df["profit"].cumsum()
    df["avg_profit"] = df["profit"].rolling(window=12 * 24).mean()

    # -----------------------------------------------------------------
    # PROFITABILITY
    # -----------------------------------------------------------------
    st.write("#### Profitability")
    st.warning(
        "**Tangible value** gained from deploying the **NeuraFlux Agent** compared to the baseline, encompassing benefits from increased **efficiency**, **cost savings**, and new **financial opportunities**."
    )

    st.write("**Key Performance Indicators**")
    _, col11, col12, col13, col14 = st.columns(
        (1, 4, 4, 4, 4), vertical_alignment="center"
    )

    # Prepare figures and data
    fig1 = plotly_filled_grad_line_chart(
        df,
        "cum_profit",
        gold_colorscale,
        line_color="gold",
        show_legend=False,
        height=350,
    )
    fig2 = plotly_colored_line_chart(
        df,
        "hourly_profit",
        show_legend=False,
        height=350,
        line_color="gold",
    )
    fig4 = create_profit_hist_plot(df["daily_profit"].dropna(), height=350)

    last_h_profit = df["profit"].iloc[-12:].sum()
    last_24h_profit = df["profit"].iloc[-12 * 24 :].sum()
    last_7_days_profit = df["profit"].iloc[-12 * 24 * 7 :].sum()
    last_30_days_profit = df["profit"].iloc[-12 * 24 * 30 :].sum()

    all_h_windows_profit = df["profit"].rolling(window=12).sum()
    all_24h_windows_profit = df["profit"].rolling(window=12 * 24).sum()
    all_7_days_windows_profit = df["profit"].rolling(window=12 * 24 * 7).sum()
    all_30_days_windows_profit = df["profit"].rolling(window=12 * 24 * 30).sum()

    delta_h = last_h_profit - all_h_windows_profit.mean()
    delta_24h = last_24h_profit - all_24h_windows_profit.mean()
    delta_7days = last_7_days_profit - all_7_days_windows_profit.mean()
    delta_30days = last_30_days_profit - all_30_days_windows_profit.mean()

    delta_h_percentage = (delta_h / abs(all_h_windows_profit.mean())) * 100
    delta_24h_percentage = (delta_24h / abs(all_24h_windows_profit.mean())) * 100
    delta_7days_percentage = (delta_7days / abs(all_7_days_windows_profit.mean())) * 100
    delta_30days_percentage = (
        delta_30days / abs(all_30_days_windows_profit.mean())
    ) * 100

    # Display Metrics
    col11.metric(
        # Center the text
        "Last hour",
        value=str(round(last_h_profit, 1)) + "$" if len(df) > 12 else "NA",
        delta=str(round(delta_h, 1)) + "%" if len(df) > 12 else "-",
        help="Percentage change from the average hourly profit.",
    )
    col12.metric(
        "Last 24h",
        value=str(round(last_24h_profit, 1)) + "$" if len(df) > 12 * 24 else "NA",
        delta=str(round(delta_24h, 1)) + "%" if len(df) > 12 * 24 else "-",
        help="Percentage change from the average daily profit.",
    )
    col13.metric(
        "Last 7 Days",
        value=str(round(last_7_days_profit, 1)) + "$"
        if len(df) > 12 * 24 * 7
        else "NA",
        delta=str(round(delta_7days, 1)) + "%" if len(df) > 12 * 24 * 7 else "-",
        help="Percentage change from the average weekly profit.",
    )
    col14.metric(
        "Last 30 Days",
        value=str(round(last_30_days_profit, 1)) + "$"
        if len(df) > 12 * 24 * 30
        else "NA",
        delta=str(round(delta_30days, 1)) + "%" if len(df) > 12 * 24 * 30 else "-",
        help="Percentage change from the average monthly profit.",
    )

    # Display Figures
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "**Cumulative**",
            "**Net Margin**",
            "**Breakdown**",
            "**Risk & Volatility**",
        ]
    )

    with tab1:
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------------------------------------------
    # FINANCIAL FLOW
    # -----------------------------------------------------------------
    st.write("#### Financial Flow")
    st.success(
        "Analysis of the **NeuraFlux Agent**'s deployment on the stakeholder's **financial dynamics**, including **expenses**, **income**, and **liquidity** impacts."
    )

    st.write("**Key Performance Indicators**")
    _, col21, col22, col23, col24 = st.columns(
        (1, 4, 4, 4, 4), vertical_alignment="center"
    )

    # Prepare figures and data
    fig1 = plotly_filled_grad_line_chart(
        df,
        "cum_cash_flow",
        green_colorscale,
        line_color="green",
        show_legend=False,
        height=350,
    )
    fig2 = plotly_colored_line_chart(
        df,
        "hourly_cash_flow",
        show_legend=False,
        height=350,
    )
    sankey_fig = plotly_sankey_plot()

    # Display Figures
    tab21, tab22, tab23, tab24 = st.tabs(
        [
            "**Cash Flow**",
            "**Transactions**",
            "**Revenue Streams**",
            "**Expenses Allocation**",
        ]
    )
    with tab21:
        st.plotly_chart(fig1, use_container_width=True)

    with tab22:
        st.plotly_chart(fig2, use_container_width=True)

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
