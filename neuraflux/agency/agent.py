import datetime as dt
import json
import logging as log
import os
from copy import copy

import numpy as np
import pandas as pd

from neuraflux.agency.control_module import ControlModule
from neuraflux.agency.control_utils import softmax
from neuraflux.agency.data_module import DataModule
from neuraflux.agency.dqn import DDQNPREstimator
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.replay_buffer import ReplayBuffer
from neuraflux.agency.time_features import tf_all_cyclic
from neuraflux.agency.utils_control import (
    convert_data_to_experience,
    convert_data_to_state,
    get_full_state_signals_from_rl_config,
)
from neuraflux.agency.utils_data import (
    add_product_data_to_df,
    add_tariff_data_to_df,
    add_vm_data_to_df,
    collect_signals_from_asset,
    cron_matches,
    get_active_config_based_on_duration,
    push_df_as_partitionned_parquet,
    read_parquet_table,
    tf_all_cyclic,
)
from neuraflux.agency.utils_registries import (
    get_entities_in_registry,
    load_dqn_estimator_from_registry,
    load_replay_buffer_from_registry,
    push_dqn_estimator_to_registry,
    push_replay_buffer_to_registry,
)
from neuraflux.agency.utils_rl_training import simple_training_loop
from neuraflux.global_variables import (
    CONTROL_KEY,
    DT_FILE_STR_FORMAT,
    DT_STR_FORMAT,
    LOG_ENTITY_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_SIM_T_KEY,
    MS_AGENT_CONTROL_DATA_KEY,
    MS_AGENT_REAL_TRAINING_KEY,
    MS_AGENT_SIM_DATA_KEY,
    MS_AGENT_SIM_TRAINING_KEY,
    MS_ASSET_SIGNAL_DATA_KEY,
    MS_SHADOW_ASSET_SIGNAL_DATA_KEY,
    TABLE_AGENT_DATA,
    TABLE_CONTROLS,
    TABLE_CONTROLS_SHADOW,
    TABLE_SIGNALS,
    TABLE_SIGNALS_SHADOW,
    TABLE_VIRTUAL_DQN_TRAINING,
    TABLE_WEATHER,
    TIMESTAMP_KEY,
)
from neuraflux.local_typing import AgentInMemoryStorageType, AssetType, UidType
from neuraflux.schemas.agency import AgentConfig, SignalTags
from neuraflux.schemas.control import DiscreteControl, PolicyEnum
from neuraflux.time_ref import TimeInfo
from neuraflux.weather import WeatherInfo


class Agent:
    """
    Represents an agent in the simulation.
    The agent is responsible for controlling an asset and interacting with the environment.
    """

    def __init__(
        self,
        uid: UidType,
        config: AgentConfig | None = None,
        directory: str = "",
        data_module: DataModule | None = None,
        control_module: ControlModule | None = None,
        # prediction_module: PredictionModule | None = None,
        time_info: TimeInfo | None = None,
    ) -> None:
        """
        Initialize the agent with the given configuration and modules.
        Args:
            uid (UidType): Unique identifier for the agent.
            config (AgentConfig | None): Configuration for the agent. Defaults to None.
            data_module (DataModule | None): Data module for data retrieval and storage. Defaults to None.
            control_module (ControlModule | None): Control module for controlling the asset. Defaults to None.
            prediction_module (PredictionModule | None): Prediction module for forecasting. Defaults to None.
            time_info (TimeInfo | None): Time information for the simulation. Defaults to None.
        """
        # Update internal attributes
        self.uid = uid
        self.config = config
        self.directory = directory
        self.data_dir = os.path.join(directory, "data")
        self.buffer_registry_dir = os.path.join(directory, "buffer_registry")
        self.dqn_registry_dir = os.path.join(directory, "dqn_registry")

        # Time reference
        if time_info is not None:
            self.update_time_info(time_info)

        # Modules
        self.data_module = data_module
        self.control_module = control_module
        # self.prediction_module = prediction_module

        # Store agent initialization time information
        self.initial_time_info = time_info
        self.elapsed_time = 0

        # Initialize agent's asset and shadow asset as None
        self.asset = None
        self.shadow_asset = None

        # Agent's local data store
        self._memory_storage: AgentInMemoryStorageType = {
            MS_AGENT_CONTROL_DATA_KEY: [],
            MS_AGENT_SIM_DATA_KEY: [],
            MS_AGENT_REAL_TRAINING_KEY: [],
            MS_AGENT_SIM_TRAINING_KEY: [],
            MS_ASSET_SIGNAL_DATA_KEY: [],
            MS_SHADOW_ASSET_SIGNAL_DATA_KEY: [],
        }

        # Create shortcuts, for convenience
        self._generate_shortcuts()

        self.epsilon = 1.0
        self.control_ready = False

    def __call__(self, *args, **kwargs):
        """
        Utility, call the agent to run its main function.
        This method is a wrapper around the run method.
        """
        return self.run(*args, **kwargs)

    def run(self) -> list[DiscreteControl] | None:
        """
        Run the agent for a given time step.
        """

        # 1. Collect asset data
        self.asset_data_collection()

        # 2. Define control and store it in memory
        if self.control_ready:
            q_factors = self.get_q_factors(use_lite_inference=False)
            if np.random.rand() <= self.epsilon:
                control = np.random.randint(3)
            else:
                control = np.argmax(q_factors[0][-1].flatten())
        control = np.random.randint(3)
        self._push_data_dict_to_memory_storage(
            storage_key=MS_AGENT_CONTROL_DATA_KEY,
            data_dict={CONTROL_KEY: control},
            timestamp=self.time_info.t,
        )

        # 3. Push in-memory data to database, and clear it
        memory_dump_freq = self.config.data.memory_dump_freq_cron
        if cron_matches(self.time_info.t, memory_dump_freq):
            self.push_in_memory_data_to_db(self.data_dir)
            self.clear_in_memory_storage()

        # 4 TODO: Train using simulated data
        
        # 5. Train using real data
        real_lr_config = get_active_config_based_on_duration(
            duration_s=self.get_elapsed_time(),
            config_dict=self.config.control.real_learning_configs,
        )
        rl_train_freq = real_lr_config.trigger_freq_cron
        if cron_matches(self.time_info.t, rl_train_freq):
            
            reward = self.get_data(start_time=self.time_info.t - dt.timedelta(days=7))[
                    "reward"
                ].sum()
            print(f"Reward in the last 7 days (eps = {self.epsilon}): {round(reward, 2)}")
            
            self.rl_training()
            self.epsilon = np.clip(round(self.epsilon - 0.1, 2), 0.0, 1.0)
            self.control_ready = True
        return [DiscreteControl(control)]

    def asset_data_collection(self):
        """
        Collect data from the asset (and shadow asset if available) and
        stores it in the local data store.
        """
        # Main asset data collection
        tracked_signals = self.config.data.tracked_signals
        asset_signals_dict = collect_signals_from_asset(
            asset=self.asset, tracked_signals=tracked_signals
        )
        # Store in memory, with associated timestamp
        self._push_data_dict_to_memory_storage(
            storage_key=MS_ASSET_SIGNAL_DATA_KEY,
            data_dict=asset_signals_dict,
            timestamp=self.time_info.t,
        )

        # Shadow asset data collection
        if self.shadow_asset is not None:
            shadow_signals_dict = collect_signals_from_asset(
                asset=self.shadow_asset, tracked_signals=tracked_signals
            )
            # Add prefix to shadow signals
            new_shadow_signals_dict = {
                f"shadow_{key}": value for key, value in shadow_signals_dict.items()
            }
            # Store in memory, with associated timestamp
            self._push_data_dict_to_memory_storage(
                storage_key=MS_SHADOW_ASSET_SIGNAL_DATA_KEY,
                data_dict=new_shadow_signals_dict,
                timestamp=self.time_info.t,
            )

    def assign_to_asset(
        self, asset: AssetType, shadow_asset: AssetType | None = None
    ) -> None:
        """
        Assign the agent to an asset.
        Args:
            asset (AssetType): The asset to assign to the agent.
            shadow_asset (AssetType | None): The shadow asset to assign to the agent. Defaults to None.
        """
        self.asset = asset
        self.shadow_asset = shadow_asset

    def clear_in_memory_storage(self, key: str | None = None) -> None:
        """
        Clear the in-memory storage for a given key.
        Args:
            key (str | None): The key to clear. If None, clear all keys.
        """
        if key is None:
            for k in self._memory_storage.keys():
                self._memory_storage[k].clear()
        else:
            self._memory_storage[key].clear()

    def get_config(self) -> AgentConfig:
        """
        Get the agent configuration.
        Returns:
            AgentConfig: The agent configuration.
        """
        return self.config

    def get_data(
        self,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        q_factors_data: bool = False,
    ):
        # Get latest data from the in-memory storage, and use datetime index
        memory_df = self.get_in_memory_data()

        if len(memory_df) > 0:
            memory_df.sort_values(by=TIMESTAMP_KEY, inplace=True)
            memory_df[TIMESTAMP_KEY] = pd.to_datetime(memory_df[TIMESTAMP_KEY])
            memory_df.set_index(TIMESTAMP_KEY, inplace=True)

            # Augment the data with additional useful columns
            df = memory_df
            df = add_vm_data_to_df(df, self.cpm)
            df = add_tariff_data_to_df(df, self.config.tariff)
            df = add_product_data_to_df(df, self.config.product)
            memory_df = tf_all_cyclic(df)
            # TODO: Add q_factors data

            # Return directly in-memory data if sufficient for user request
            if start_time is not None and start_time in memory_df.index:
                memory_df = memory_df.loc[memory_df.index >= start_time]
                if end_time is not None:
                    memory_df = memory_df.loc[memory_df.index <= end_time]
                return memory_df

        # Get longer-term data from the database
        long_term_data = self.get_database_data()
        if len(memory_df) == 0:
            df = long_term_data
        else:
            memory_df.reset_index(inplace=True)
            df = pd.concat([long_term_data, df], axis=0, ignore_index=True)

        # Make sure output index is with a datetime index
        df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY])
        df.set_index(TIMESTAMP_KEY, inplace=True)

        # Apply time filters, if submitted
        if start_time is not None:
            df = df.loc[df.index >= start_time]
        if end_time is not None:
            df = df.loc[df.index < end_time]

        return df

    def get_database_data(self):
        """
        Get the data from the local database.
        Returns:
            pd.DataFrame: The data from the database.
        """
        return read_parquet_table(self.data_dir)

    def get_elapsed_time(self) -> float:
        """
        Get the elapsed time since the agent initialization.
        Returns:
            float: The elapsed time in seconds.
        """
        return (self.time_info.t - self.initial_time_info.t).total_seconds()

    def get_in_memory_data(
        self,
        keys: list[str] | None = None,
        join: bool = True,
    ) -> dict[str, pd.DataFrame] | pd.DataFrame:
        """
        Get the in-memory data for the agent.
        Args:
            keys (list[str] | None): The keys to retrieve. If None, retrieve all keys.
            join (bool): Whether to join the data into a single DataFrame, using timestamp. Defaults to True.
        Returns:
            dict[str, pd.DataFrame] | pd.DataFrame: The in-memory data.
        """
        # Check if keys are provided, if not, use all keys
        if keys is None:
            keys = self._memory_storage.keys()

        # Check if keys are valid
        data = {}
        for key in keys:
            data[key] = pd.DataFrame(self._memory_storage[key])

        # Return dict if no join is requested
        if not join:
            return data

        # Otherwise ...
        # 1. Filter and prepare the DataFrames that contain the "timestamp" column.
        dfs = [
            df.set_index(TIMESTAMP_KEY)
            for df in data.values()
            if not df.empty and TIMESTAMP_KEY in df.columns
        ]
        # 2. Concatenate them along the columns using an outer join.
        if len(dfs) == 0:
            return pd.DataFrame()
        final_df = pd.concat(dfs, axis=1, join="outer")
        # 3. Reset the index and sort the DataFrame
        final_df.reset_index(inplace=True)
        final_df.sort_values(by=TIMESTAMP_KEY, inplace=True)

        return final_df

    def get_q_estimators_list(self, registry_dir: str | None = None) -> list[str]:
        """
        Get the list of Q estimators for the agent, sorted by creation time.

        Args:
            registry_dir (str | None): The directory for the Q estimator registry. Defaults to None,
                which uses the agent's directory.
        Returns:
            list[str]: The list of Q estimators for the agent.
        """
        registry_dir = self.dqn_registry_dir if registry_dir is None else registry_dir
        available_estimators = get_entities_in_registry(registry_dir)
        agent_available_estimators = sorted(
            [e for e in available_estimators if e.startswith(self.uid)]
        )
        return agent_available_estimators

    def get_q_estimator(
        self, registry_dir: str | None = None, name: str | None = None
    ) -> tuple[DDQNPREstimator, dict]:
        """
        Get the Q estimator for the agent.

        Args:
            registry_dir (str | None): The directory for the Q estimator registry. Defaults to None,
                which uses the agent's directory.
            name (str | None): The name of the Q estimator. If None, the latest estimator is used, or
                a new one is created if none exist.
        Returns:
            tuple[DDQNPREstimator, dict]: The Q estimator for the agent and its metadata.
        """
        registry_dir = self.dqn_registry_dir if registry_dir is None else registry_dir
        # Case 1 - User directly specified the name of the estimator
        if name is not None:
            return load_dqn_estimator_from_registry(registry_dir, name=name)

        agent_available_estimators = self.get_q_estimators_list(
            registry_dir=registry_dir
        )

        # Case 2 - User did not specify the name of the estimator, but some are available
        if agent_available_estimators:
            latest_estimator = agent_available_estimators[-1]
            return load_dqn_estimator_from_registry(registry_dir, name=latest_estimator)

        # Case 3 - No estimators available, create a new one
        rl_config = self.config.control.rl_config
        state_columns = get_full_state_signals_from_rl_config(rl_config)
        estimator = DDQNPREstimator(
            state_size=len(state_columns),
            action_size=rl_config.action_size,
            sequence_len=rl_config.history_length,
            n_controllers=self.config.control.n_controllers,
            # NOTE: Other entries will be overwritten at fit time
        )
        return estimator, {}

    def get_q_factors(self, use_lite_inference: bool = True) -> np.ndarray:
        # Get the data just for current time-step
        rl_config = self.config.control.rl_config
        rl_seq_len = rl_config.history_length
        delta_required = self.time_info.dt.total_seconds() * (rl_seq_len - 1)
        start_time = self.time_info.t - dt.timedelta(seconds=delta_required)
        df = self.get_data(start_time=start_time)
        # df = self.get_data(start)

        state_columns = get_full_state_signals_from_rl_config(rl_config)
        q_estimator, _ = self.get_q_estimator(self.dqn_registry_dir)
        states = np.array(convert_data_to_state(df, state_columns, rl_seq_len))

        q_factors = q_estimator.forward_pass(states)

        return q_factors

    def get_replay_buffer(
        self, simulation: bool = False, registry_dir: str | None = None
    ) -> tuple[ReplayBuffer, dict]:
        """
        Get the replay buffer for the agent.

        Args:
            simulation (bool): Whether to get the simulated replay buffer. Defaults to False.
            registry_dir (str | None): The directory for the replay buffer registry. Defaults to None.
        Returns:
            tuple[ReplayBuffer, dict]: The replay buffer for the agent and its metadata.
        """
        # Define variables related to buffer
        registry_dir = (
            self.buffer_registry_dir if registry_dir is None else registry_dir
        )
        buffer_name = self.uid + "_sim" if simulation else self.uid + "_real"

        # Try to retrieve the replay buffer from the registry if it exists
        available_buffers = get_entities_in_registry(registry_dir)
        if buffer_name in available_buffers:
            buffer, buffer_metadata = load_replay_buffer_from_registry(
                registry_dir, name=buffer_name
            )

        # Otherwise, initialize a new replay buffer
        else:
            buffer_len = (
                self.config.control.sim_replay_buffer_size
                if simulation
                else self.config.control.real_replay_buffer_size
            )
            buffer = ReplayBuffer(
                max_len=buffer_len,
                prioritized_replay_alpha=0.6,
            )
            buffer_metadata = {
                "capacity": buffer_len,
                "n_experiences": 0,
            }
        return buffer, buffer_metadata

    def get_uid(self) -> UidType:
        """
        Get the unique identifier of the agent.
        Returns:
            UidType: The unique identifier of the agent.
        """
        return self.uid

    def push_in_memory_data_to_db(self, storage_path: str) -> None:
        # Get the data from the in-memory storage
        df = self.get_in_memory_data()

        # Make the timestamp column a datetime index
        df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY])
        df.set_index(TIMESTAMP_KEY, inplace=True)

        # Augment the data
        df = add_vm_data_to_df(df, self.cpm)
        df = add_tariff_data_to_df(df, self.config.tariff)
        df = add_product_data_to_df(df, self.config.product)
        df = tf_all_cyclic(df)

        # Convert the index to a column
        df.reset_index(inplace=True)

        # Save to local storage
        push_df_as_partitionned_parquet(df, storage_path)

    def push_data_to_replay_buffer(
        self,
        data: pd.DataFrame,
        registry_dir: str,
        simulation: bool = False,
    ):
        # Get agent's replay buffer (real or simulated)
        replay_buffer, replay_buffer_metadata = self.get_replay_buffer(
            simulation=simulation, registry_dir=registry_dir
        )

        # Define important quantities to transform data into experience
        rl_config = self.config.control.rl_config
        state_columns = rl_config.state_signals
        state_columns = get_full_state_signals_from_rl_config(rl_config)
        control_columns = [col for col in data.columns if col.startswith(CONTROL_KEY)]
        seq_len = rl_config.history_length
        product = AvailableProductsEnum.from_string(self.config.product)
        reward_columns = product.get_reward_names()

        # Convert data to RL experience
        experience_batch = convert_data_to_experience(
            data, seq_len, state_columns, control_columns, reward_columns
        )

        # Add the experience samples from the dataframe to the replay buffer
        for experience in zip(*experience_batch):
            replay_buffer.add_experience_sample(experience=experience)

        # Save the replay buffer to the registry
        buffer_name = self.uid + "_sim" if simulation else self.uid + "_real"
        replay_buffer_metadata["n_experiences"] = len(replay_buffer)
        print(
            f"Replay buffer now has {replay_buffer_metadata['n_experiences']} samples."
        )
        push_replay_buffer_to_registry(
            registry_dir=registry_dir,
            name=buffer_name,
            replay_buffer=replay_buffer,
            metadata=replay_buffer_metadata,
            overwrite=True,
        )

    def rl_training(self) -> None:
        # Define start time based on previous trainings (all if first time)
        start_time = None
        if hasattr(self, "last_rl_training_end"):
            start_time = self.last_rl_training_end

        # Get data, and keep track of last index to avoid future overlaps
        history = self.get_data(start_time=start_time)
        self.last_rl_training_end = history.index[-1]

        # Update scaler with dataframe
        # self.data_module.update_scaler_info_from_df(self.uid, df=history)

        # Remove rows with NaN values
        n_samples_before = history.shape[0]
        history = history.dropna()
        n_samples_after = history.shape[0]
        # If any rows were removed, raise an error
        if n_samples_after != n_samples_before:
            raise ValueError("NaN values detected in training data.")

        # Add new data in the replay buffer
        self.push_data_to_replay_buffer(
            data=history, registry_dir=self.buffer_registry_dir
        )

        # Retrieve buffer and q-estimator from registry
        buffer, buffer_metadata = self.get_replay_buffer(registry_dir=self.buffer_registry_dir)
        q_estimator, _ = self.get_q_estimator(registry_dir=self.dqn_registry_dir)

        # Training loop
        for _ in range(20):
            q_estimator, buffer, _ = simple_training_loop(
                replay_buffer=buffer,
                q_estimator=q_estimator,
            )
            q_estimator.update_target_model()

        # Save new Q-estimator to registry
        estimator_name = self.uid + "_" + self.time_info.t.strftime(DT_FILE_STR_FORMAT)
        estimator_metadata = {"last_training": self.time_info.get_t_as_str()}
        push_dqn_estimator_to_registry(
            registry_dir=self.dqn_registry_dir,
            name=estimator_name,
            estimator=q_estimator,
            metadata=estimator_metadata,
        )

        # Save the replay buffer to the registry
        buffer_name = self.uid + "_real"
        buffer_metadata["n_experiences"] = len(buffer)
        push_replay_buffer_to_registry(
            registry_dir=self.buffer_registry_dir,
            name=buffer_name,
            replay_buffer=buffer,
            metadata=buffer_metadata,
            overwrite=True,
        )

        del buffer, q_estimator, estimator_metadata, history

    def update_time_info(self, time_info: TimeInfo) -> None:
        """
        Update the time information of the agent.
        Args:
            time_info (TimeInfo): The time information to update.
        """
        self.time_info = time_info
        self._generate_shortcuts()

    def save_config(self, filepath: str) -> None:
        """
        Save the agent configuration to a JSON file.
        Args:
            filepath (str): The path to the JSON file.
        """
        # Dump agent configuration to JSON file
        with open(filepath, "w") as f:
            config_json = self.config.model_dump()
            json.dump(config_json, f, indent=4)

    def _generate_shortcuts(self):
        self.cpm = self.config.data.control_power_mapping
        self.t = self.time_info.t

    def _push_data_dict_to_memory_storage(
        self,
        storage_key: str,
        data_dict: dict[str, list[str | float | int]],
        timestamp: dt.datetime = None,
    ) -> None:
        """
        Push data to the in-memory storage.
        Args:
            storage_key (str): The key to store the data under.
            data_dict (dict[str, list]): The data to push.
            timestamp (dt.datetime | None): The timestamp to associate with the data. Defaults to None. Defaults to True.
        """
        # Create a copy of the data dictionary
        data_dict = copy(data_dict)

        # Add current timestamp to the data dictionary if required
        if timestamp is not None:
            data_dict[TIMESTAMP_KEY] = timestamp

        # Append the data to the in-memory storage
        self._memory_storage[storage_key].append(data_dict)


class OldAgent:
    def __init__(
        self,
        uid: UidType,
        config: AgentConfig,
        data_module: DataModule,
        control_module: ControlModule,
        # prediction_module: PredictionModule,
        time_info: TimeInfo,
        weather_info: WeatherInfo,
    ) -> None:
        # Update internal attributes
        self.uid = uid
        self.config = config
        self.update_time_info(time_info)
        self.update_weather_info(weather_info)
        self.update_modules(
            data_module=data_module,
            control_module=control_module,
            # prediction_module=prediction_module,
        )

        # Initialize agent attributes
        self.control_ready = False

        # Store agent initialization time information
        self.initial_time_info = time_info

        # Shortcuts
        self.cpm = self.config.data.control_power_mapping

        # Initialize agent's asset and shadow asset as None
        self.asset = None
        self.shadow_asset = None

        self.traj_counter = 0
        self.epsilon = 1.0

    # -----------------------------------------------------------------------
    # DATA RETRIEVAL
    # -----------------------------------------------------------------------
    def get_data(
        self,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        df = self.data_module.get_agent_data(
            self.uid,
            start_time=start_time,
            end_time=end_time,
            control_power_mapping=self.cpm,
            tariff=self.config.tariff,
            product=self.config.product,
            **kwargs,
        )

        # Log
        log.debug(
            {
                LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "get_data",
                LOG_MESSAGE_KEY: f"Retrieved data from {start_time} to {end_time} with kwargs {str(kwargs)}.",
            }
        )

        return df

    def get_data_for_rl_at_time(
        self, time_idx: None | dt.datetime = None
    ) -> pd.DataFrame:
        # Use current time index if unspecified
        if time_idx is None:
            time_idx = self.time_info.t

        # Define delta time required for RL data, based on history length
        rl_seq_len = self.real_rl_config.history_length
        delta_seconds_required = self.time_info.dt.total_seconds() * rl_seq_len

        # Verify agent has enough data
        if time_idx < self.initial_time_info.t + dt.timedelta(
            seconds=delta_seconds_required
        ):
            raise ValueError("Not enough data available for RL.")

        # Get data for RL
        start_time = time_idx - dt.timedelta(seconds=delta_seconds_required)
        df = self.get_data(start_time=start_time, time_features=True)
        return df

    def get_q_factors(
        self, rl_df: None | pd.DataFrame = None, use_lite_inference: bool = False
    ) -> np.ndarray:
        rl_df = self.get_data_for_rl_at_time() if rl_df is None else rl_df

        # Keep only the necessary history length to have a batch of 1
        if use_lite_inference:
            rl_df = rl_df.iloc[-self.real_rl_config.history_length :]

        # Scale the dataframe
        # scaled_rl_df = self.data_module.scale_dataframe_from_scaling_dict(
        #    rl_df, self.uid
        # )
        q_factors = self.control_module.get_raw_q_factors(
            self.uid,
            rl_df,
            self.data_module.get_columns_with_tag(self.config, SignalTags.RL_STATE),
            len(list(self.cpm.keys())),
            self.real_rl_config,
            use_lite_inference=use_lite_inference,
        )

        # Delete unused variables and force garbage collection
        # del rl_df, scaled_rl_df
        del rl_df

        return q_factors

    # TODO: replace with get_data method
    def get_reward_data(self, *args, **kwargs) -> pd.DataFrame:
        # Work with a copy of the data
        data = self.get_data(*args, **kwargs).copy()

        # Get reward column name(s)
        product_str = self.config.product
        product = AvailableProductsEnum.from_string(product_str)
        reward_columns = product.get_reward_names()
        data[reward_columns] = product.calculate_rewards(data)

        # Delete unused variables and force garbage collection
        del product

        return data[reward_columns]

    def get_rl_training_data_table(self):
        rl_training_data = self.control_module.get_rl_training_data_table(self.uid)
        return rl_training_data

    def get_training_summary(self):
        training_summary = self.control_module.get_training_summary(self.uid)
        return training_summary

    # -----------------------------------------------------------------------
    # CONTROL & POLICIES
    # -----------------------------------------------------------------------
    def get_controls(
        self,
        policy: PolicyEnum = PolicyEnum.Q_MAX,
        policy_kwargs: dict = {},
        store_controls: bool = False,
    ) -> dict[str, DiscreteControl] | None:
        # Return controls based on specified policy, if agent is ready
        if self.control_ready:
            match policy:
                case PolicyEnum.Q_MAX:
                    policy_fn = self.apply_policy_q_max
                case PolicyEnum.RANDOM:
                    policy_fn = self.apply_policy_random
            control_data = policy_fn(**policy_kwargs)
        else:
            # control_data = self.apply_policy_random()
            # Use asset's default if not ready
            controls_list = self.asset.get_auto_control(
                timestamp=self.time_info.t,
                outside_air_temperature=self.weather_info.temperature,
            )
            control_data = {
                CONTROL_KEY + "_" + str(i + 1): c.value
                for i, c in enumerate(controls_list)
            }

        # Store controls generated to the database if desired
        if store_controls:
            self._push_control_data_to_db(control_data, self.time_info.t)

        # Return None if no or missing controls are generated by the policy
        if any([c is None for c in control_data.values()]):
            return None

        # Sanitize control data to return
        sanitized_controls = {k: DiscreteControl(v) for k, v in control_data.items()}

        # Log
        log.debug(
            {
                LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "get_controls",
                LOG_MESSAGE_KEY: f"get_controls with {policy.value} policy returning "
                f"{str({k: v.value for k, v in sanitized_controls.items()})} "
                f"with kwargs {str(policy_kwargs)}",
            }
        )

        return sanitized_controls

    def apply_policy_random(self) -> dict[str, int]:
        n_controllers = self.config.control.n_controllers
        action_size = len(list(self.cpm.keys()))
        control_data = {}
        for c in range(n_controllers):
            control_key = CONTROL_KEY + "_" + str(c + 1)
            control_data[control_key] = np.random.randint(action_size)

        # Log
        log.debug(
            {
                LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "apply_policy_random",
                LOG_MESSAGE_KEY: f"Random policy returning controls {str(control_data)}",
            }
        )

        return control_data

    def apply_policy_q_max(
        self,
        rl_df: None | pd.DataFrame = None,
        epsilon: float | None = None,
        min_probability: float = 0.05,
        boltzmann: bool = False,
        use_lite_inference: bool = True,
    ) -> dict[str, int]:
        # Initialize necessary variables and quantities
        q_factors = self.get_q_factors(rl_df, use_lite_inference=use_lite_inference)
        possible_actions = list(self.cpm.keys())
        control_data = {}

        # Define epsilon for epsilon-greedy policy
        if epsilon is None:
            epsilon = self.epsilon

            # training_summary = self.control_module.get_training_summary(self.uid)
            # n_trainings = (
            #    0
            #    if "n_trainings" not in training_summary
            #    else training_summary["n_trainings"]
            # )
            # epsilon = (20 - n_trainings) / 20 if epsilon is None else epsilon

        epsilon = np.clip(epsilon, 0, 1)
        # Loop over controllers and generate control data
        for c, q_values in enumerate(q_factors):  # c = controller
            control_key = CONTROL_KEY + "_" + str(c + 1)
            # Random action probability
            if np.random.rand() < epsilon:  # Random action
                control_data[control_key] = int(np.random.choice(possible_actions))
                policy_output = "epsilon random"
            # Else, greedy action
            else:
                q_values = q_values.flatten()
                # Boltzmann sampling mode
                if boltzmann:
                    softmax_probabilities = softmax(q_values)
                    # Ensure each probability is at least min_probability
                    probabilities = np.maximum(softmax_probabilities, min_probability)

                    # Normalize the probabilities to sum to 1
                    probabilities /= np.sum(probabilities)
                    control_int = np.random.choice(possible_actions, p=probabilities)
                    control_data[control_key] = int(control_int)
                    policy_output = "boltzmann sampling"
                # Classical argmax mode
                else:
                    control_data[control_key] = int(np.argmax(q_values))
                    policy_output = "argmax q-values"

            df_in = self.get_data_for_rl_at_time()

            # Log
            log.debug(
                {
                    LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                    LOG_ENTITY_KEY: f"Agent({self.uid})",
                    LOG_METHOD_KEY: "apply_policy_q_max",
                    LOG_MESSAGE_KEY: f"Q-Max policy for controller {c + 1} (eps={epsilon:.2f}) using {policy_output} returning controls {str(control_data)}, based on {q_factors} obtained with inputs {df_in}",
                }
            )

        # log.debug(f"Q-Max policy with epsilon={epsilon} returning {control_data}")

        # Delete unused variables and force garbage collection
        del rl_df, q_factors, df_in, q_values

        return control_data

    # -----------------------------------------------------------------------
    # LEARNING AND TRAINING
    # -----------------------------------------------------------------------
    def rl_training(self) -> None:
        # ADD REAL DATA TO REPLAY BUFFER
        # Define start time based on previous trainings
        start_time = None
        if hasattr(self, "last_rl_training_end"):
            start_time = self.last_rl_training_end
        # history = self.data_module.get_agent_data(
        #     self.uid,
        #     start_time=start_time,
        #     end_time=None,
        #     control_power_mapping=self.cpm,
        #     tariff=self.config.tariff,
        #     product=self.config.product,
        #     time_features=True,
        # )
        history = self.get_data(start_time=start_time, time_features=True)
        # Keep track of the last training end time to avoid overlapping data
        self.last_rl_training_end = history.index[-1]

        # Update scaler with dataframe
        self.data_module.update_scaler_info_from_df(self.uid, df=history)

        # Remove rows with NaN values
        n_samples_before = history.shape[0]
        history = history.dropna()
        n_samples_after = history.shape[0]
        # If more than 20% of time is removed, raise an error
        if (n_samples_after / n_samples_before) < 0.8:
            raise ValueError("Too many NaN values in training data")

        # Push real data to replay buffer
        self.data_module.store_agent_data_in_table(
            df=history, uid=self.uid, table_name=TABLE_AGENT_DATA
        )
        self.push_data_as_rl_training_data(history)

        # ADD SIMULATED DATA TO REPLAY BUFFER
        # TODO: Add this once simulation is available again
        # Loop over all trajectories, at each time index
        # for idx in history.index:
        #    trajectories = self.data_module.get_trajectories(self.uid, idx)
        #    for trajectory in trajectories:
        #        print(trajectory)
        #        print()
        #        self.push_data_as_rl_training_data(trajectory, simulation=True)

        # PERFORM REINFORCEMENT LEARNING TRAINING
        # Get state columns from the data module
        state_columns = self.data_module.get_columns_with_tag(
            self.config, SignalTags.RL_STATE
        )
        action_size = len(list(self.cpm.keys()))
        self.control_module.rl_training(
            self.uid,
            self.real_rl_config,
            self.time_info.t,
            self.config.product,
            state_columns,
            action_size,
        )

        # Agent is now ready to perform control
        self.control_ready = True

        # Delete unused variables and force garbage collection
        del history, state_columns

        self.epsilon -= 0.1

    def simulated_rl_training(
        self,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        n_timesteps_to_sample: int | None = 100,
        training_name: str = "daily_training",
    ) -> None:
        # Variables def
        history_len = self.sim_rl_config.history_length
        n_traj_samples = (
            self.config.control.sim_training_config.n_trajectory_samples_per_timestep
        )
        traj_len = self.config.control.sim_training_config.trajectory_len

        # Define start time based on previous trainings
        if hasattr(self, "last_sim_rl_training_end"):
            start_time = self.last_sim_rl_training_end - (
                self.time_info.dt * history_len
            )

        # Download necessary data
        df = self.get_data(start_time=start_time, end_time=end_time, time_features=True)

        self.last_sim_rl_training_end = df.index[-1]

        print(
            f"    Simulating {n_timesteps_to_sample} trajectories in time ({n_traj_samples} per timestep) for training from {df.index[0]} to {df.index[-1]}"
        )

        # Loop over all indexes if n_samples is None, else sample uniform
        samples_batch = (
            df.index[:-history_len]
            if n_timesteps_to_sample is None
            else np.random.choice(
                df.index[: -(history_len + traj_len)],
                n_timesteps_to_sample,
                replace=False,
            )
        )

        for t in samples_batch:
            for _ in range(n_traj_samples):
                self.traj_counter += 1
                # Get the data for the current time
                df_0 = df.loc[df.index >= t].iloc[:history_len]

                trajectory = self.sample_trajectory(
                    df_0,
                    traj_len,
                    # policy=self.apply_policy_random,
                    # policy_kwargs={},
                    # model=None,
                )

                # Push simulated data to replay buffer
                self.push_data_as_rl_training_data(trajectory, simulation=True)

                # Augment data with additional columns for future analytics
                trajectory["model"] = "exact_asset_model"
                trajectory["policy"] = "random"
                trajectory["exec_time"] = dt.datetime.now(dt.UTC).isoformat()
                trajectory["train_batch"] = training_name
                trajectory["relative_index"] = range(-history_len + 1, traj_len + 1)
                trajectory["is_real"] = [True] * history_len + [False] * traj_len
                trajectory["traj_id"] = self.traj_counter

                # Store each row in the database, itterating over rows as dicts
                for idx, row in trajectory.iterrows():
                    self.data_module.store_data_in_table_at_time(
                        self.uid,
                        TABLE_VIRTUAL_DQN_TRAINING + training_name,
                        idx,
                        row.to_dict(),
                        data_columns=list(trajectory.columns),
                    )

        # PERFORM REINFORCEMENT LEARNING TRAINING
        # Get state columns from the data module
        state_columns = self.data_module.get_columns_with_tag(
            self.config, SignalTags.RL_STATE
        )
        action_size = len(list(self.cpm.keys()))
        self.control_module.rl_training(
            self.uid,
            self.sim_rl_config,
            self.time_info.t,
            self.config.product,
            state_columns,
            action_size,
            simulation=True,
        )

        del df, state_columns

    def sample_trajectory(
        self,
        df_0: pd.DataFrame,
        trajectory_len: int,
        # policy: PolicyEnum,
        # policy_kwargs: dict,
        # model: callable,
    ) -> pd.DataFrame:
        # Work with a copy of the input dataframe
        df = df_0.copy()
        control_cols = [
            CONTROL_KEY + "_" + str(i + 1) for i in range(self.asset.config.n_controls)
        ]
        index = df_0.index[-1]

        # Define asset copy
        asset_copy = self.return_asset_copy_from_snapshot(index)

        # Preload weather data
        final_index = index + dt.timedelta(minutes=5 * trajectory_len)
        weather_df = self.data_module.get_data_from_table(
            self.uid, TABLE_WEATHER, index + dt.timedelta(minutes=5), final_index
        )
        df = pd.concat([df, weather_df])

        # Create trajectory sample
        current_idx = index
        for _ in range(trajectory_len):
            next_idx = current_idx + dt.timedelta(minutes=5)

            # Augment before to make sure policy has all it needs
            df = self.asset.augment_df(df)
            df = self.data_module.augment_dataframe_with_virtual_metering_data(
                df, self.cpm
            )
            df = self.data_module.augment_dataframe_with_tariff_data(
                df, self.config.tariff
            )
            df = self.data_module.augment_dataframe_with_product_data(
                df, self.config.product
            )
            df = tf_all_cyclic(df)

            rl_seq_len = self.sim_rl_config.history_length
            rl_df = df.loc[:current_idx].iloc[-rl_seq_len:]

            #########
            if self.control_ready:
                # control_values = list(policy(**policy_kwargs).values())
                control_values = self.apply_policy_q_max(
                    rl_df, epsilon=0.5, boltzmann=True, use_lite_inference=True
                )
            else:
                control_values = self.apply_policy_random()

            df.loc[current_idx, control_cols] = control_values
            df[control_cols] = df[control_cols].astype("Int64")

            df = self.asset.augment_df(df)
            df = self.data_module.augment_dataframe_with_virtual_metering_data(
                df, self.cpm
            )
            df = self.data_module.augment_dataframe_with_tariff_data(
                df, self.config.tariff
            )
            df = self.data_module.augment_dataframe_with_product_data(
                df, self.config.product
            )
            df = tf_all_cyclic(df)

            # Asset signals collection
            control_cols = [
                CONTROL_KEY + "_" + str(i + 1)
                for i in range(self.asset.config.n_controls)
            ]
            control_values_from_df = df.loc[current_idx, control_cols].values
            controls_list = [DiscreteControl(int(c)) for c in control_values_from_df]
            asset_copy.step(
                control=controls_list,
                timestamp=current_idx,
                outside_air_temperature=df.loc[current_idx, "outside_air_temperature"],
            )
            signals_dict = {}
            for signal in self.config.data.tracked_signals:
                signal_value = asset_copy.get_signal(signal)

                # Store array-like values separately
                if isinstance(signal_value, (list, tuple, np.ndarray)):
                    for i, value in enumerate(signal_value):
                        signals_dict[signal + "_" + str(i + 1)] = value
                else:
                    signals_dict[signal] = signal_value
            for k, v in signals_dict.items():
                df.loc[next_idx, k] = v
            # df = self.prediction_module.get_model_prediction(
            #     self.uid, df, current_idx, next_idx
            # )
            # End prediction

            current_idx = next_idx

        # Update one last time df to avoid NaNs in next state
        df = self.asset.augment_df(df)
        df = self.data_module.augment_dataframe_with_virtual_metering_data(df, self.cpm)
        df = self.data_module.augment_dataframe_with_tariff_data(df, self.config.tariff)
        df = self.data_module.augment_dataframe_with_product_data(
            df, self.config.product
        )
        df = tf_all_cyclic(df)

        return df

    def push_data_as_rl_training_data(
        self, data: pd.DataFrame, simulation: bool = False
    ):
        # Work with a copy of the data
        data = data.copy()

        # Add reward and done columns
        product_str = self.config.product
        product = AvailableProductsEnum.from_string(product_str)
        reward_columns = product.get_reward_names()
        data[reward_columns] = product.calculate_rewards(data)
        data = product.calculate_dones(data)

        # Get control columns
        control_columns = [col for col in data.columns if col.startswith(CONTROL_KEY)]

        # State columns
        state_columns = self.data_module.get_columns_with_tag(
            self.config, SignalTags.RL_STATE
        )

        # Scale the data
        # scaled_data = self.data_module.scale_dataframe_from_scaling_dict(data, self.uid)
        # TODO Add generalized check for this type of circumstance
        # scaled_data = scaled_data.iloc[:-1]

        # Define RL config based on simulation or real training
        rl_config = self.real_rl_config if not simulation else self.sim_rl_config
        action_size = len(list(self.cpm.keys()))

        # Convert and push data to the replay buffers
        self.control_module.push_data_to_replay_buffer(
            uid=self.uid,
            data=data,
            rl_config=rl_config,
            state_columns=state_columns,
            action_size=action_size,
            control_columns=control_columns,
            reward_columns=reward_columns,
            simulation=simulation,
        )

        # Delete unused variables and force garbage collection
        # del data, scaled_data, product, control_columns, reward_columns
        del data, product, control_columns, reward_columns

    # -----------------------------------------------------------------------
    # UTILITIES AND SIMULATION INTERACTIONS
    # -----------------------------------------------------------------------
    def asset_data_collection(self) -> None:
        # Asset signals collection
        signals_dict = {}
        for signal in self.config.data.tracked_signals:
            signal_value = self.asset.get_signal(signal)

            # Store array-like values separately
            if isinstance(signal_value, (list, tuple, np.ndarray)):
                for i, value in enumerate(signal_value):
                    signals_dict[signal + "_" + str(i + 1)] = value
            else:
                signals_dict[signal] = signal_value

        # Shadow asset signals collection
        if self.shadow_asset is not None:
            shadow_signals_dict = {}
            for signal in self.config.data.tracked_signals:
                signal_value = self.shadow_asset.get_signal(signal)

                # Store array-like values separately
                if isinstance(signal_value, (list, tuple, np.ndarray)):
                    for i, value in enumerate(signal_value):
                        shadow_signals_dict[signal + "_" + str(i + 1)] = value
                else:
                    shadow_signals_dict[signal] = signal_value

        # Weather Data Definition
        weather_dict = {
            "outside_air_temperature": self.weather_info.temperature,
        }

        # Storedata in Agent's database
        self._push_asset_signal_data_to_db(signals_dict, self.time_info.t)
        if self.shadow_asset is not None:
            self._push_asset_signal_data_to_db(
                shadow_signals_dict, self.time_info.t, shadow_asset=True
            )
        self._push_weather_data_to_db(weather_dict, self.time_info.t)

        # Log
        formatted_weather_dict = {
            key: f"{value:.2f}" if isinstance(value, float) else value
            for key, value in weather_dict.items()
        }
        formatted_asset_signals_dict = {
            key: f"{value:.2f}" if isinstance(value, float) else value
            for key, value in signals_dict.items()
        }
        log.debug(
            {
                LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "asset_data_collection",
                LOG_MESSAGE_KEY: f"Stored weather data: {str(formatted_weather_dict)}.",
            }
        )
        log.debug(
            {
                LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "asset_data_collection",
                LOG_MESSAGE_KEY: f"Sampled following signals from asset: {str(formatted_asset_signals_dict)}.",
            }
        )

        # Delete unused variables and force garbage collection
        del signals_dict, weather_dict

    def update_weather_info(self, weather_info: WeatherInfo) -> None:
        self.weather_info = weather_info

    # -----------------------------------------------------------------------
    # ASSET-BASED METHODS
    # -----------------------------------------------------------------------
    def return_asset_copy_from_snapshot(self, timestamp: dt.datetime) -> AssetType:
        AssetClass = self.asset.__class__
        asset_config = copy(self.asset.config)
        initial_state_dict = copy(asset_config.initial_state_dict)
        data = self.get_data(start_time=timestamp, end_time=timestamp)

        new_initial_state_dict = self._reverse_initial_state(initial_state_dict, data)
        asset_config.initial_state_dict = new_initial_state_dict
        oat = data["outside_air_temperature"].values[0]
        return AssetClass("temp_copy", asset_config, timestamp, oat)

    def _reverse_initial_state(self, init_variables, df):
        # Reconstruct the initial state dictionary
        new_state_dict = {}
        for variable in init_variables:
            # Check if the variable is supposed to be an array/list
            if isinstance(init_variables[variable], (list, tuple, np.ndarray)):
                # Find all columns in the dataframe that start with this variable name and a suffix
                suffix_columns = [
                    col for col in df.columns if col.startswith(variable + "_")
                ]
                # Extract the values for these columns and form a list or array
                if len(suffix_columns) > 0:
                    values = [df[col].iloc[0] for col in sorted(suffix_columns)]
                    # Check the type to maintain consistency with the original
                    if isinstance(init_variables[variable], list):
                        new_state_dict[variable] = values
                    elif isinstance(init_variables[variable], tuple):
                        new_state_dict[variable] = tuple(values)
                    elif isinstance(init_variables[variable], np.ndarray):
                        new_state_dict[variable] = np.array(values)
            else:
                # For non-array types, directly take the value from the dataframe
                if variable in df.columns:
                    new_state_dict[variable] = df[variable].iloc[0]

        return new_state_dict

    # -----------------------------------------------------------------------
    # PRIVATE METHODS
    # -----------------------------------------------------------------------
    def _push_asset_signal_data_to_db(
        self,
        signals_dict: dict[str, float | int | str],
        timestamp: dt.datetime,
        shadow_asset: bool = False,
    ) -> None:
        signal_names_list = self.config.data.tracked_signals
        table_name = TABLE_SIGNALS_SHADOW if shadow_asset else TABLE_SIGNALS
        self.data_module.store_data_in_table_at_time(
            self.uid,
            table_name,
            timestamp,
            signals_dict,
            data_columns=signal_names_list,
        )

        # Delete unused variables and force garbage collection
        del signals_dict, signal_names_list

    def _push_weather_data_to_db(
        self, weather_dict: dict[str, float | int | str], timestamp: dt.datetime
    ) -> None:
        self.data_module.store_data_in_table_at_time(
            self.uid,
            TABLE_WEATHER,
            timestamp,
            weather_dict,
            data_columns=list(weather_dict.keys()),
        )

        # Delete unused variables and force garbage collection
        del weather_dict

    def _push_control_data_to_db(
        self,
        control_dict: dict[str, int],
        timestamp: dt.datetime,
        shadow_asset: bool = False,
    ) -> None:
        control_keys = [
            CONTROL_KEY + "_" + str(c + 1)
            for c in range(self.config.control.n_controllers)
        ]
        table_name = TABLE_CONTROLS_SHADOW if shadow_asset else TABLE_CONTROLS
        self.data_module.store_data_in_table_at_time(
            self.uid,
            table_name,
            timestamp,
            control_dict,
            data_columns=control_keys,
        )

        # Delete unused variables and force garbage collection
        del control_dict, control_keys
