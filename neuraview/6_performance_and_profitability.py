import streamlit as st
import datetime as dt
import pandas as pd
import os
from plot_utils import plotly_filled_grad_line_chart
from sidebar_utils import generate_sidebar
from streamlit import session_state as ss

generate_sidebar()

if "agent" in ss:
    if "df" not in ss:
        ss.df = ss.agent.get_data(q_factors=True)
        ss.df["cum_reward"] = ss.df["reward"].cumsum()

    fig = plotly_filled_grad_line_chart(ss.df, "cum_reward", "Blues", "blue")
    st.plotly_chart(fig, use_container_width=True)
    st.write(ss.df[ss.df.index > dt.datetime(2023, 1, 23)])
