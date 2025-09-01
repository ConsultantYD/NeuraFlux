from neuraflux.agency.agent import Agent
import os

SIM_DIR = "simulations"
CASE_STUDY = "case_study_1"
AGENT_UID = "Agent001"
agent_dir = os.path.join(SIM_DIR, CASE_STUDY, AGENT_UID)
agent = Agent.from_dir(agent_dir)


