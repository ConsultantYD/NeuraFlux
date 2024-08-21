import datetime as dt
import json
import logging as log
import os
import random
import shutil
from copy import deepcopy

import numpy as np
import tensorflow as tf

from neuraflux.agency.agent import Agent
from neuraflux.agency.control_module import ControlModule
from neuraflux.agency.data_module import DataModule
from neuraflux.assets.building import Building
from neuraflux.assets.electric_vehicle import ElectricVehicle
from neuraflux.assets.energy_storage import EnergyStorage
from neuraflux.global_variables import (
    DT_STR_FORMAT,
    LOG_ENTITY_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_SIM_T_KEY,
    LOGGING_DB_NAME,
)
from neuraflux.local_typing import AssetType
from neuraflux.logging_utils import StructuredLogHandler
from neuraflux.schemas.asset_config import (
    BuildingConfig,
    ElectricVehicleConfig,
    EnergyStorageConfig,
)
from neuraflux.schemas.simulation import SimulationConfig
from neuraflux.time_ref import TimeRef
from neuraflux.weather import Weather


class Simulation:
    def __init__(self, simulation_config: SimulationConfig) -> None:
        # Initialize key internal attributes
        self.config = simulation_config
        self.directory = simulation_config.directory

        # Create simulation directory if it does not exist
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # Initialize logging and related configurations
        log_dir = self.config.directory
        if os.path.isfile(os.path.join(log_dir, LOGGING_DB_NAME)):
            os.remove(os.path.join(log_dir, LOGGING_DB_NAME))
        handler = StructuredLogHandler(log_dir)
        log.getLogger().addHandler(handler)
        log.getLogger().setLevel(log.DEBUG)

        # Fix seeds for reproducibility
        self._fix_seeds(self.config.seed)

        # Temporal evolution initialization
        self.start_time = dt.datetime.strptime(
            self.config.time.start_time,
            DT_STR_FORMAT,
        )
        self.end_time = dt.datetime.strptime(
            self.config.time.end_time,
            DT_STR_FORMAT,
        )
        self._initialize_time()

        # Weather initialization
        self.weather_ref = Weather(
            city=self.config.geography.city, db_dir=self.directory
        )
        t = self.time_info.t
        self.weather_info = self.weather_ref.get_weather_info_at_time(t)

        # Initialize assets and comparative shadow assets
        self.assets = self._initialize_assets()
        self.shadow_assets = self._initialize_shadow_assets()

        # Initialize agency modules
        self._initialize_modules()

        # Initialize agents
        self._initialize_agents()

    def _fix_seeds(self, seed_value: int) -> None:
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

    def run(self) -> None:
        # Save simulation summary before starting
        self.sim_summary = {
            "time start": self.start_time.strftime(DT_STR_FORMAT),
            "time end": self.end_time.strftime(DT_STR_FORMAT),
            "current time": self.time_info.t.strftime(DT_STR_FORMAT),
        }
        with open(os.path.join(self.directory, "sim_summary.json"), "w") as f:
            json.dump(self.sim_summary, f, indent=4)

        # Save configuration to directory as well (can be reproduced)
        with open(os.path.join(self.directory, "config.json"), "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)

        while self.time_info.t <= self.end_time:  # Main time loop
            self.sim_summary["current time"] = self.time_info.t.strftime(DT_STR_FORMAT)
            with open(os.path.join(self.directory, "sim_summary.json"), "w") as f:
                json.dump(self.sim_summary, f, indent=4)

            # Calculate elapsed time in minutes since simulation start
            delta_since_start = self.time_ref.get_time_delta_since_start()
            elapsed_minutes = delta_since_start.total_seconds() / 60

            # Loop over agents
            agents_controls = {}
            for uid, agent in self.agents.items():
                agent.update_time_info(self.time_info)
                agent.update_weather_info(self.weather_info)

                # Sample data from the asset associated to the agent
                agent.asset_data_collection()

                # Get agent controls
                control_dict = agent.get_controls(store_controls=True)
                agents_controls[uid] = (
                    None if control_dict is None else list(control_dict.values())
                )

            # Update simulation time
            self.time_ref.increment_time()

            # Update time and weather info
            self.time_info = self.time_ref.get_time_info()
            self.weather_info = self.weather_ref.get_weather_info_at_time(
                self.time_info.t
            )

            # Time info
            log.info(
                {
                    LOG_SIM_T_KEY: self.time_info.t,
                    LOG_ENTITY_KEY: "Simulation",
                    LOG_METHOD_KEY: "run",
                    LOG_MESSAGE_KEY: f"~~~~~~~~~~~~~~~~ {self.time_info.t} ~~~~~~~~~~~~~~~~",
                }
            )

            # Loop over assets
            for uid, agent in self.agents.items():
                asset = self.assets[uid]
                shadow_asset = self.shadow_assets[uid]

                # Advance asset simulation, using agent control if available
                if agents_controls[uid] is not None:
                    asset.step(
                        agents_controls[uid],
                        self.time_info.t,
                        self.weather_info.temperature,
                    )
                else:
                    asset.auto_step(self.time_info.t, self.weather_info.temperature)

                # Keep shadow asset in sync with the real asset for comparison
                shadow_asset.auto_step(self.time_info.t, self.weather_info.temperature)

            # Agent update/training loop
            if elapsed_minutes % (60 * 24 * 7) == 0 and elapsed_minutes != 0:
                for uid, agent in self.agents.items():
                    reward = agent.get_reward_data()
                    # Print only last 24h
                    reward_last_24h = reward.iloc[-288:]
                    print(
                        f"Agent {uid} reward (sum|mean|max): {round(reward_last_24h.sum(axis=0).iloc[0],4)}|{round(reward_last_24h.mean().iloc[0],4)}|{round(reward_last_24h.max().iloc[0],4)}"
                    )
                    time_train_starts = dt.datetime.now(dt.UTC)
                    agent.rl_training()
                    time_train_ends = dt.datetime.now(dt.UTC)
                    print(
                        f"Training time (s): {(time_train_ends - time_train_starts).total_seconds()}"
                    )

    def save(self) -> None:
        raise NotImplementedError

    def load(self, sim_dir=None) -> None:
        raise NotImplementedError

    @classmethod
    def from_directory(cls, sim_dir: str) -> None:
        raise NotImplementedError

    def _initialize_time(self) -> None:
        self.time_ref = TimeRef(
            start_time_utc=self.start_time,
            def_time_step=dt.timedelta(seconds=self.config.time.step_size_s),
        )
        self.time_info = self.time_ref.get_time_info()

        # Time info
        log.info(
            {
                LOG_SIM_T_KEY: self.time_info.t,
                LOG_ENTITY_KEY: "Simulation",
                LOG_METHOD_KEY: "run",
                LOG_MESSAGE_KEY: f"~~~~~~~~~~~~~~~~ {self.time_info.t} ~~~~~~~~~~~~~~~~",
            }
        )

    def _initialize_modules(self) -> None:
        self.data_module = DataModule(base_dir=self.directory)
        self.control_module = ControlModule(base_dir=self.directory)

    def _initialize_assets(self) -> dict[str, AssetType]:
        assets: dict[str, AssetType] = {}
        ext_temperature = self.weather_info.temperature
        for asset_uid, asset_config_dict in self.config.assets.items():
            # INITIAL STATE DEFINITION
            # Add outside air temperature to asset config
            asset_config_dict.initial_state_dict["outside_air_temperature"] = (
                ext_temperature
            )

            # ASSET INSTANTIATION
            asset_type = asset_config_dict.asset_type
            match asset_type:
                case "building":
                    asset_config = BuildingConfig.model_validate(asset_config_dict)
                    asset = Building(
                        asset_uid, asset_config, self.time_info.t, ext_temperature
                    )
                case "electric vehicle":
                    asset_config = ElectricVehicleConfig.model_validate(
                        asset_config_dict
                    )
                    asset = ElectricVehicle(
                        asset_uid, asset_config, self.time_info.t, ext_temperature
                    )
                case "energy storage":
                    asset_config = EnergyStorageConfig.model_validate(asset_config_dict)
                    asset = EnergyStorage(
                        asset_uid, asset_config, self.time_info.t, ext_temperature
                    )
                case _:
                    raise ValueError(f"Asset type not supported: {asset_type}.")

            assets[asset_uid] = asset

        return assets

    def _initialize_shadow_assets(self) -> dict[str, AssetType]:
        # Initialize shadow assets as an exact copy of the real assets
        shadow_assets = {uid: deepcopy(asset) for uid, asset in self.assets.items()}

        # Modify asset name to differentiate from real asset
        for asset in shadow_assets.values():
            asset.name = f"{asset.name}_shadow"

        return shadow_assets

    def _initialize_agents(self) -> None:
        self.agents: dict[str, Agent] = {}

        # Loop over agents
        for uid, agent_config in self.config.agents.items():
            # Initialize agent directory
            agent_dir = os.path.join(self.directory, uid)
            if os.path.isdir(agent_dir):
                shutil.rmtree(agent_dir)  # Delete if exists
            os.makedirs(agent_dir)

            # Propagation configuration elements for None values
            # TODO: Replace this to avoid config fields redundancy
            if agent_config.control.reinforcement_learning.n_controllers is None:
                n_controllers = agent_config.control.n_controllers
                agent_config.control.reinforcement_learning.n_controllers = (
                    n_controllers
                )

            # Initialize new agent instance
            agent = Agent(
                uid=uid,
                time_info=self.time_info,
                config=agent_config,
                data_module=self.data_module,
                control_module=self.control_module,
                weather_info=self.weather_info,
            )

            # Save Agent config in directory
            config_filepath = os.path.join(agent_dir, "config.json")
            agent.save_config(config_filepath)

            # Assign Agent instance to its corresponding asset and shadow asset
            agent.assign_to_asset(self.assets[uid])
            agent.assign_to_shadow_asset(self.shadow_assets[uid])

            # Initialize infrastructure for agent
            self.data_module.initialize_new_agent_data_infrastructure(
                uid, agent_config.data.signals_info
            )

            # Add agent to simulation
            self.agents[uid] = agent
