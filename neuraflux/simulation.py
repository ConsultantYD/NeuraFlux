import datetime as dt
import os
import random
import shutil
from copy import deepcopy

import numpy as np
import tensorflow as tf

from neuraflux.agency.agent import Agent
from neuraflux.agency.control_module import ControlModule
from neuraflux.agency.data_module import DataModule
from neuraflux.assets.factory import AvailableAssetsEnum
from neuraflux.geography import CityEnum
from neuraflux.global_variables import (
    CONTROL_KEY,
    DT_STR_FORMAT,
    OAT_KEY,
    TIMESTAMP_KEY,
)
from neuraflux.local_typing import AssetType
from neuraflux.schemas.agency import AgentConfig
from neuraflux.schemas.simulation import SimulationConfig
from neuraflux.time_ref import TimeRef
from neuraflux.weather import Weather
from neuraflux.agency.utils_data import (
    add_vm_data_to_df,
    add_tariff_data_to_df,
    add_product_data_to_df,
    push_df_as_partitionned_parquet,
    cron_matches,
    read_parquet_table,
    tf_all_cyclic,
)


class Simulation:
    def __init__(self, simulation_config: SimulationConfig) -> None:
        # Initialize key internal attributes
        self.config = simulation_config
        self.directory = simulation_config.directory

        # Create simulation directory if it does not exist
        os.makedirs(self.directory, exist_ok=True)

        # Fix seeds for reproducibility
        self._fix_seeds(self.config.seed)

        # ---------------------------------------------------
        # - ENVIRONMENT
        # ---------------------------------------------------
        # Time-related components initialization
        self.sim_start_time = dt.datetime.strptime(
            self.config.time.start_time,
            DT_STR_FORMAT,
        )
        self.sim_end_time = dt.datetime.strptime(
            self.config.time.end_time,
            DT_STR_FORMAT,
        )
        self.time_ref = self._initialize_time_reference(
            start_time=self.sim_start_time, step_size_s=self.config.time.step_size_s
        )
        self.time_info = self.time_ref.get_time_info()
        self.t = self.time_info.t

        # Weather
        self.weather_ref = self._initialize_weather(
            city=self.config.geography.city, db_dir=self.directory
        )
        self.weather_info = self.weather_ref.get_weather_info_at_time(self.t)
        self.oat = self.weather_info.temperature

        # ---------------------------------------------------
        # - ASSETS
        # ---------------------------------------------------
        # Initialize assets controlled by agents
        self.assets = self._initialize_assets(
            t=self.t,
            oat=self.weather_info.temperature,
            assets_configs_dict=self.config.assets,
        )

        # Initialize comparative 'shadow' assets (copy of real assets)
        self.shadow_assets = self._initialize_shadow_assets(
            real_assets_dict=self.assets,
        )

        # ---------------------------------------------------
        # - AGENTS AND RELATED COMPONENTS
        # ---------------------------------------------------
        # Initialize agents
        self.agents = self._initialize_agents(
            agent_configs_dict=self.config.agents,
            directory=self.directory,
            time_info=self.time_info,
            assets=self.assets,
            shadow_assets=self.shadow_assets,
        )

    def run(self) -> None:
        # Save simulation summary before starting
        # self.sim_summary = {
        #     "time start": self.start_time.strftime(DT_STR_FORMAT),
        #     "time end": self.end_time.strftime(DT_STR_FORMAT),
        #     "current time": self.time_info.t.strftime(DT_STR_FORMAT),
        # }
        # with open(os.path.join(self.directory, "sim_summary.json"), "w") as f:
        #     json.dump(self.sim_summary, f, indent=4)

        # # Save configuration to directory as well (can be reproduced)
        # with open(os.path.join(self.directory, "config.json"), "w") as f:
        #     json.dump(self.config.model_dump(), f, indent=4)

        # Main simulation
        while self.time_info.t < self.sim_end_time:
            # ---------------------------------------------------
            # - AGENTS EXECUTION
            # ---------------------------------------------------
            # Main agent action loop
            agents_controls = {}
            for uid, agent in self.agents.items():
                # Update agent time referential info, and run it
                agent.update_time_info(self.time_info)
                agents_controls[uid] = agent.run()

            # Increment simulation by one time step
            # NOTE: Increment is done here to allow agents to run on initial state
            self.time_ref.increment_time()

            # Update time and weather info for simulation
            self.time_info = self.time_ref.get_time_info()
            self.t = self.time_info.t
            self.weather_info = self.weather_ref.get_weather_info_at_time(self.t)
            self.oat = self.weather_info.temperature

            # Loop over assets
            for uid, agent in self.agents.items():
                asset = self.assets[uid]
                shadow_asset = self.shadow_assets[uid]

                # Advance asset simulation, using agent control if available
                if agents_controls[uid] is not None:
                    asset.step(
                        agents_controls[uid],
                        self.t,
                        self.oat,
                    )
                # Use default asset policy if no agent control specified
                else:
                    asset.auto_step(self.t, self.oat)

                # Keep shadow asset in sync with the real asset for comparison
                shadow_control = shadow_asset.get_auto_control(self.t, self.oat)
                shadow_control_dict = {}
                for c in range(len(shadow_control)):
                    control_key = CONTROL_KEY + "_" + str(c + 1)
                    shadow_control_dict[control_key] = shadow_control[c].value
                shadow_asset.step(shadow_control, self.t, self.oat)

            # Save agent's full data to disk when requested
            if cron_matches(
                self.time_info.t,
                self.config.agent_save_freq_cron,
            ):
                # Save agents' data to disk
                for uid, agent in self.agents.items():
                    agent_directory = os.path.join(
                        self.directory,
                        uid,
                    )
                    agent.to_file(directory=agent_directory)

        for uid in self.agents.keys():
            agent_dir = os.path.join(self.directory, uid)
            new_agent = Agent.from_dir(agent_dir)
            shadow_asset = new_agent.shadow_asset
            df = shadow_asset.get_historical_data()
            df = add_vm_data_to_df(df, agent.cpm)
            df = add_tariff_data_to_df(df, agent.config.tariff)
            df = add_product_data_to_df(df, agent.config.product)

            # Rename all columns with a "shadow_" prefix
            df = df.rename(columns={col: f"shadow_{col}" for col in df.columns})

            print(df)

    def _fix_seeds(self, seed_value: int) -> None:
        """
        Fixes the random seed for reproducibility. Covers numpy, random, and TensorFlow.
        Args:
            seed_value (int): The seed value to set for random number generation.
        """
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

    def _initialize_time_reference(
        self, start_time: dt.datetime, step_size_s: int
    ) -> TimeRef:
        """
        Initializes the time reference for the simulation. Sets the start time, end time, and time step size.
        Args:
            start_time (dt.datetime): The start time of the simulation.
            step_size_s (int): The time step size in seconds.
        Returns:
            TimeRef: An instance of the TimeRef class, which encapsulates time-related information.
        """
        return TimeRef(
            start_time_utc=start_time,
            def_time_step=dt.timedelta(seconds=step_size_s),
        )

    def _initialize_weather(self, city: CityEnum, db_dir: str) -> Weather:
        """
        Initializes the weather reference for the simulation.
        Args:
            city (CityEnum): The city for which the weather information is required.
            db_dir (str): The directory where the weather data will be stored.
        """
        return Weather(city=city, db_dir=db_dir)

    def _initialize_modules(self, directory: str) -> list[ControlModule, DataModule]:
        """
        Initializes the agency modules for the whole simulation.
        Args:
            directory (str): The directory where the modules will be stored.
        Returns:
            list[ControlModule, DataModule]: A list containing the control and data modules.
        """
        control_module = ControlModule(base_dir=directory)
        data_module = DataModule(base_dir=directory)
        return control_module, data_module

    def _initialize_assets(
        self, t: dt.datetime, oat: float, assets_configs_dict: dict[str, object]
    ) -> dict[str, AssetType]:
        """
        Initializes the assets for the simulation based on the configuration provided.
        Args:
            t (dt.datetime): The current time of the simulation.
            oat (float): The outside air temperature, in DegC.
            assets_configs_dict (dict[str, object]): A dictionary containing the asset configurations.
        Returns:
            dict[str, AssetType]: A dictionary mapping asset UIDs to their respective asset instances.
        """
        # Loop over all inputed asset configs
        assets: dict[str, AssetType] = {}
        for asset_uid, asset_config_dict in assets_configs_dict.items():
            # Initial state definition
            asset_config_dict.initial_state_dict[OAT_KEY] = oat

            # Asset instances creation
            asset_type = asset_config_dict.asset_type
            AssetClass = AvailableAssetsEnum.get_asset_class_from_asset_name(asset_type)
            AssetConfigClass = (
                AvailableAssetsEnum.get_asset_config_class_from_asset_name(asset_type)
            )
            asset = AssetClass(
                asset_uid,
                AssetConfigClass.model_validate(asset_config_dict),
                t,
                oat,
            )
            assets[asset_uid] = asset

        return assets

    def _initialize_shadow_assets(
        self, real_assets_dict: dict[str, AssetType]
    ) -> dict[str, AssetType]:
        """
        Initializes shadow assets as an exact copy of the real assets.
        Args:
            real_assets_dict (dict[str, AssetType]): A dictionary mapping asset UIDs to their respective asset instances.
        Returns:
            dict[str, AssetType]: A dictionary mapping asset UIDs to their respective shadow asset instances.
        """
        # Initialize shadow assets as an exact copy of the real assets
        shadow_assets = {
            uid: deepcopy(asset) for uid, asset in real_assets_dict.items()
        }

        # Modify asset name to differentiate from real asset
        for asset in shadow_assets.values():
            asset.name = f"{asset.name}_shadow"

        return shadow_assets

    def _initialize_agents(
        self,
        directory: str,
        time_info: TimeRef,
        agent_configs_dict: dict[str, AgentConfig],
        assets: dict[str, AssetType],
        shadow_assets: dict[str, AssetType],
    ) -> dict[str, Agent]:
        """
        Initializes the agents for the simulation based on the configuration provided.
        Args:
            directory (str): The directory where the agents will be stored.
            agent_configs_dict (dict[str, AgentConfig]): A dictionary containing the agent configurations.
            assets (dict[str, AssetType]): A dictionary mapping asset UIDs to their respective asset instances.
            shadow_assets (dict[str, AssetType]): A dictionary mapping asset UIDs to their respective shadow asset instances.
        Returns:
            dict[str, Agent]: A dictionary mapping agent UIDs to their respective agent instances.
        """
        agents: dict[str, Agent] = {}

        # Loop over all input agent configs
        for uid, agent_config in agent_configs_dict.items():
            # Initialize agent directory
            agent_dir = os.path.join(directory, uid)
            if os.path.isdir(agent_dir):
                shutil.rmtree(agent_dir)  # Delete if exists
            os.makedirs(agent_dir)

            # Initialize agent instance
            agent = Agent(
                uid=uid,
                directory=agent_dir,
                time_info=time_info,
                config=agent_config,
                data_module=None,
                control_module=None,
            )

            # Save Agent config in directory
            config_filepath = os.path.join(agent_dir, "config.json")
            agent.save_config(config_filepath)

            # Assign Agent instance to its corresponding asset and shadow asset
            agent.assign_to_asset(assets[uid], shadow_assets[uid])

            # Add agent to simulation
            agents[uid] = agent
        return agents
