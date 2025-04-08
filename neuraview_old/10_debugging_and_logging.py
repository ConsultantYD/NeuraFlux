import streamlit as st
from sidebar_utils import generate_sidebar


# Page title and sidebar initialization
st.header("🩺 Debugging & Diagnostics")
generate_sidebar()

st.write("#### Session State Viewer")
col01, _ = st.columns(2)
all_available_keys = list(st.session_state.keys())
all_available_keys.sort()

key_to_view = col01.selectbox("Select a key to view", all_available_keys, index=None)

if key_to_view is not None:
    st.write(st.session_state[key_to_view])
