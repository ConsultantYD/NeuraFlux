import streamlit as st
import pandas as pd
import os 
from plot_utils import plotly_filled_grad_line_chart

SIM_DIR = "simulations/simple_validation"
AGENT = "Agent001"

df = pd.read_parquet(os.path.join(SIM_DIR, AGENT, "data"))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

st.dataframe(df)

df["cum_reward"] = df["reward"].cumsum()

fig = plotly_filled_grad_line_chart(
    df, "cum_reward", "Blues", "blue"
)
st.plotly_chart(fig, use_container_width=True)