import streamlit as st
import datetime as dt
import pandas as pd
import os
from plot_utils import (
    plotly_filled_grad_line_chart,
    plotly_colored_line_chart,
    create_profit_hist_plot,
    plotly_sankey_plot,
    create_radar_plot,
)
from sidebar_utils import generate_sidebar
from streamlit import session_state as ss
from neuraflux.agency.utils_data import (
    add_vm_data_to_df,
    add_tariff_data_to_df,
    add_product_data_to_df,
)

generate_sidebar()

if "agent" in ss:
    if "shadow_df" not in ss:
        # Get dataframe for shadow asset, to use in comparison
        shadow_asset = ss.agent.shadow_asset
        df = shadow_asset.get_historical_data()
        df = add_vm_data_to_df(df, ss.agent.cpm)
        df = add_tariff_data_to_df(df, ss.agent.config.tariff)
        df = add_product_data_to_df(df, ss.agent.config.product)
        ss.shadow_df = df.rename(columns={col: f"shadow_{col}" for col in df.columns})

    if "df" not in ss:
        ss.df = ss.agent.get_data(q_factors=True)

        # Join with shadow_df
        ss.df = ss.df.join(ss.shadow_df, how="outer")

    df = ss.df

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

    df["profit"] = df["shadow_price_$"] - df["price_$"]
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
    with st.expander("**Key Performance Indicators**", expanded=False):
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
        height=300,
    )
    fig2 = plotly_colored_line_chart(
        df,
        "hourly_profit",
        show_legend=False,
        height=300,
        line_color="gold",
    )
    fig4 = create_profit_hist_plot(df["daily_profit"].dropna(), height=300)

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
    tab1, tab2, tab3 = st.tabs(
        [
            "**Cumulative**",
            "**Net Margin**",
            "**Risk & Volatility**",
        ]
    )

    with tab1:
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.plotly_chart(fig4, use_container_width=True)

        # -----------------------------------------------------------------
    # REWARD PLOTS
    # -----------------------------------------------------------------
    st.write("##### Reward Signals")
    st.info(
        "**Reinforcement signals** generated and **maximized** by the **NeuraFlux Agent**, focusing on their **alignment** with stakeholder **objectives** and their **accrual** over different time spans."
    )

    fig3 = plotly_filled_grad_line_chart(
        df,
        "cum_reward",
        blue_colorscale,
        line_color="blue",
        show_legend=False,
        height=300,
    )

    fig4 = plotly_colored_line_chart(
        df, "avg_reward", line_color="blue", show_legend=False, height=300
    )

    # Display Figures
    tab21, tab22 = st.tabs(
        [
            "**Cumulative**",
            "**Instantaneous**",
        ]
    )
    with tab21:
        st.plotly_chart(fig3, use_container_width=True)
    with tab22:
        st.plotly_chart(fig4, use_container_width=True)

    # fig = plotly_filled_grad_line_chart(ss.df, "cum_reward", "Blues", "blue")
    # st.plotly_chart(fig, use_container_width=True)
    # st.write(ss.df[ss.df.index > dt.datetime(2023, 1, 23)])

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
        height=300,
    )
    fig2 = plotly_colored_line_chart(
        df,
        "hourly_cash_flow",
        show_legend=False,
        height=300,
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
