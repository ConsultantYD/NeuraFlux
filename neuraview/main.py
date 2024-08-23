import streamlit as st

pg = st.navigation(
    [
        st.Page("1_agent_profile.py", title="Agent Profile", icon="♟️"),
        st.Page("2_decision_engine.py", title="Decision Engine", icon="🧠"),
        st.Page("3_grid_and_power_flow.py", title="Energy Grid Dynamics", icon="⚡️"),
        st.Page("4_learning_and_evolution.py", title="Learning & Evolution", icon="🧬"),
        st.Page(
            "5_multiagent_and_aggregation.py",
            title="Multi-Agent Aggregation",
            icon="🧩",
        ),
        st.Page(
            "6_performance_and_profitability.py",
            title="Performance & Profitability",
            icon="🚀",
        ),
        st.Page(
            "7_health_and_integrity.py", title="Resilience & Reliability", icon="🛡️"
        ),
        st.Page("8_simulation_manager.py", title="Simulation Manager", icon="🏔️"),
        st.Page("9_simulation_config.py", title="Simulation Settings", icon="⚙️"),
        st.Page(
            "10_debugging_and_logging.py", title="Debugging & Diagnostics", icon="🩺"
        ),
    ]
)
pg.run()
