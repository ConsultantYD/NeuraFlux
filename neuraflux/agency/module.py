import os

import dill

from neuraflux.schemas.agency import AgentConfig
from neuraflux.local_typing import UidType


class Module:
    """
    A base class for all modules in the NeuraFlux framework.

    Methods:
        initialize_new_agent: Initializes a new agent in the system.
        from_file: Loads the module from a file.
        to_file: Saves the module to a file.
    """

    def __init__(self, base_dir: str = ""):
        self.base_dir = base_dir
        self.agents: dict[str, AgentConfig] = {}

    def initialize_new_agent(self, uid: UidType, agent_config: AgentConfig) -> None:
        """
        Initializes a new agent by creating its database and tables.

        Args:
            uid (str): The unique identifier for the agent.
            agent_config (AgentConfig): The configuration for the agent.
        """
        # Check if the agent already exists
        if uid in self.agents:
            raise ValueError(f"Agent with UID {uid} already exists.")
        self.agents[uid] = agent_config

    @classmethod
    def from_file(cls, directory: str = "") -> "Module":
        """
        Loads the module from a file.
        Args:
            directory (str): The directory to load the file from. If not provided, uses the base directory.
        Returns:
            Module: An instance of the module.
        """
        filepath = os.path.join(directory, cls.__name__)
        with open(filepath, "rb") as f:
            instance = dill.load(f)
        return instance

    def to_file(self, directory: str = "") -> None:
        """
        Saves the module to a file.

        Args:
            directory (str): The directory to save the file in. If not provided, uses the base directory.

        Returns:
            None
        """
        filepath = os.path.join(directory, self.__class__.__name__)
        with open(filepath, "wb") as f:
            dill.dump(self, f)
