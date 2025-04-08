import streamlit as st
import plotly

from data_acquisition import get_agent_and_modules

st.header("Reinforcement Learning Training")

# Constants
AGENT_UID = "Agent001"
SIM_NAME = "Sim2"

# Get agent and module objects
agent, data_module, control_module = get_agent_and_modules(AGENT_UID, SIM_NAME)

# agent_df = agent.get_data(time_features=True)
training_df = control_module.get_training_data(AGENT_UID)

# Show distribution plots evolving for each target_iterator value
st.subheader("Distribution Plots")
