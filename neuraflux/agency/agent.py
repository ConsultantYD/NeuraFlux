import datetime as dt
import json
import os
from copy import copy
import dill

import numpy as np
import pandas as pd

from neuraflux.agency.control_module import ControlModule
from neuraflux.agency.data_module import DataModule
from neuraflux.agency.dqn import DDQNPREstimator
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.replay_buffer import ReplayBuffer
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
    MS_AGENT_CONTROL_DATA_KEY,
    MS_AGENT_REAL_TRAINING_KEY,
    MS_AGENT_SIM_DATA_KEY,
    MS_AGENT_SIM_TRAINING_KEY,
    MS_ASSET_SIGNAL_DATA_KEY,
    TIMESTAMP_KEY,
)
from neuraflux.local_typing import AgentInMemoryStorageType, AssetType, UidType
from neuraflux.schemas.agency import AgentConfig
from neuraflux.schemas.control import DiscreteControl
from neuraflux.time_ref import TimeInfo


class Agent:
    """
    Represents an agent in the simulation.
    The agent is responsible for controlling an asset and interacting with the environment.
    """

    def __init__(
        self,
        uid: UidType | None = None,
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
        # Return directly if uid is None (used for loading from file)
        if uid is None:
            return

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
        action_size = self.config.control.rl_config.action_size
        n_controllers = self.config.control.n_controllers
        if self.control_ready:
            q_factors = self.get_q_factors(use_lite_inference=False)
            if np.random.rand() <= self.epsilon:
                control = [
                    int(np.random.randint(0, action_size)) for _ in range(n_controllers)
                ]
            else:
                control = [
                    int(np.argmax(q_factors[c][-1].flatten()))
                    for c in range(n_controllers)
                ]
        else:
            control = [
                int(np.random.randint(0, action_size)) for _ in range(n_controllers)
            ]
        self._push_data_dict_to_memory_storage(
            storage_key=MS_AGENT_CONTROL_DATA_KEY,
            data_dict={
                CONTROL_KEY + f"_{i+1}": control[i] for i in range(len(control))
            },
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
        if real_lr_config.enabled and cron_matches(self.time_info.t, rl_train_freq):
            reward = self.get_data(start_time=self.time_info.t - dt.timedelta(days=1))[
                "reward"
            ].sum()
            print(
                f"Reward in the last 1 day (eps = {self.epsilon}): {round(reward, 2)}"
            )

            self.rl_training()
            self.epsilon = np.clip(round(self.epsilon - 0.1, 2), 0.0, 1.0)
            self.control_ready = True
        return [DiscreteControl(c) for c in control]

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

    @classmethod
    def from_dir(cls, agent_dir: str) -> None:
        """
        Load an agent from a directory.
        Args:
            agent_dir (str): The directory to load the agent from.
        Returns:
            Agent: The loaded agent.
        """
        pickle_path = os.path.join(agent_dir, "agent.pkl")
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                internal_dict = dill.load(f)
            agent = cls()
            agent.__dict__.update(internal_dict)
            return agent
        raise ValueError(
            f"Agent not found in directory {agent_dir}. Please check the path."
        )

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
        q_factors: bool = False,
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

        # Add Q-factors data if requested
        if q_factors:
            q_factors = self.get_q_factors(df)

            seq_len = self.config.control.rl_config.history_length
            action_size = self.config.control.rl_config.action_size
            n_controllers = self.config.control.n_controllers
            n_rewards = 1

            for controller in range(n_controllers):
                for r in range(n_rewards):
                    q_cols = [f"C{controller+1}R{r+1}Q{j}" for j in range(action_size)]
                    df.loc[df.index[seq_len - 1 :], q_cols] = q_factors[controller][
                        :, r, :
                    ]

        # Make sure output index is with a datetime index
        df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY])
        df.set_index(TIMESTAMP_KEY, inplace=True)

        # Apply time filters, if submitted
        if start_time is not None:
            df = df.loc[df.index >= start_time]
        if end_time is not None:
            df = df.loc[df.index < end_time]

        return df

    def get_rl_data_for_timestamp(
        self, timestamp: dt.datetime | None = None, timestep_s: int = 300
    ) -> pd.DataFrame:
        """
        Get the data for the RL agent at a specific time step.
        Args:
            timestamp (dt.datetime | None): The timestamp to get data for. Defaults to None.
            timestep_s (int): The time step in seconds. Defaults to 300.
        Returns:
            pd.DataFrame: The data for the RL agent.
        """
        # Get the data just for current time-step
        if timestamp is None:
            timestamp = self.time_info.t
        rl_config = self.config.control.rl_config
        rl_seq_len = rl_config.history_length
        delta_required = timestep_s * (rl_seq_len - 1)
        start_time = timestamp - dt.timedelta(seconds=delta_required)
        end_time = timestamp + dt.timedelta(seconds=timestep_s)
        df = self.get_data(start_time=start_time, end_time=end_time)
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

    def get_q_factors(
        self, df: pd.DataFrame | None = None, use_lite_inference: bool = True
    ) -> list[np.ndarray]:
        """
        Get the Q factors for the agent.
        Args:
            df (pd.DataFrame | None): The data to use for the Q factors. Defaults to None.
            use_lite_inference (bool): Whether to use lite inference. Defaults to True.
        Returns:
            np.ndarray: The Q factors for the agent.
        """
        rl_config = self.config.control.rl_config
        rl_seq_len = rl_config.history_length
        if df is None:
            df = self.get_rl_data_for_timestamp()

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
        buffer, buffer_metadata = self.get_replay_buffer(
            registry_dir=self.buffer_registry_dir
        )
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

    def to_file(self, directory: str = "") -> None:
        """
        Save the agent's state to a file using dill.
        Args:
            directory (str): The directory to save the file in.
        """
        filepath = os.path.join(directory, "agent.pkl")
        internal_dict = vars(self)
        with open(filepath, "wb") as f:
            dill.dump(internal_dict, f)

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
        if hasattr(self, "config"):
            self.cpm = self.config.data.control_power_mapping
        if hasattr(self, "time_info"):
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
