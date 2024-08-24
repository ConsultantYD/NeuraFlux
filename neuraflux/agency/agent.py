import datetime as dt
import json
import logging as log

import numpy as np
import pandas as pd

from neuraflux.agency.control_module import ControlModule
from neuraflux.agency.control_utils import softmax
from neuraflux.agency.data_module import DataModule
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.global_variables import (
    CONTROL_KEY,
    LOG_ENTITY_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_SIM_T_KEY,
    TABLE_CONTROLS,
    TABLE_CONTROLS_SHADOW,
    TABLE_SIGNALS,
    TABLE_SIGNALS_SHADOW,
    TABLE_WEATHER,
)
from neuraflux.local_typing import AssetType, UidType
from neuraflux.schemas.agency import AgentConfig, SignalTags
from neuraflux.schemas.control import DiscreteControl, PolicyEnum
from neuraflux.time_ref import TimeInfo
from neuraflux.weather import WeatherInfo


class Agent:
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
        rl_seq_len = self.rl_config.history_length
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

    def get_q_factors(self, rl_df: None | pd.DataFrame = None) -> np.ndarray:
        rl_df = self.get_data_for_rl_at_time() if rl_df is None else rl_df
        scaled_rl_df = self.data_module.scale_dataframe_from_scaling_dict(
            rl_df, self.uid
        )
        q_factors = self.control_module.get_raw_q_factors(
            self.uid,
            scaled_rl_df,
            self.data_module.get_columns_with_tag(self.config, SignalTags.RL_STATE),
            self.rl_config,
        )

        # Delete unused variables and force garbage collection
        del rl_df, scaled_rl_df

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
            # TODO: Replace random policy with reading default asset one
            control_data = self.apply_policy_random()

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
                f"{str({k:v.value for k,v in sanitized_controls.items()})} "
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
    ) -> dict[str, int]:
        # Initialize necessary variables and quantities
        q_factors = self.get_q_factors(rl_df)
        possible_actions = list(self.cpm.keys())
        control_data = {}
        training_summary = self.control_module.get_training_summary(self.uid)
        n_trainings = (
            0
            if "n_trainings" not in training_summary
            else training_summary["n_trainings"]
        )

        # Define epsilon for epsilon-greedy policy
        epsilon = (20 - n_trainings) / 20 if epsilon is None else epsilon
        epsilon = np.clip(epsilon, 0, 1)

        # Loop over controllers and generate control data
        for c, q_tensor in enumerate(q_factors):  # c = controller
            # Take mean of all objectives if there are multiple
            q_values = np.mean(q_tensor[c, :, :], axis=0)
            control_key = CONTROL_KEY + "_" + str(c + 1)
            # Random action probability
            if np.random.rand() < epsilon:  # Random action
                control_data[control_key] = int(np.random.choice(possible_actions))
                policy_output = "epsilon random"
            # Else, greedy action
            else:
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
            # Log
            log.debug(
                {
                    LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                    LOG_ENTITY_KEY: f"Agent({self.uid})",
                    LOG_METHOD_KEY: "apply_policy_q_max",
                    LOG_MESSAGE_KEY: f"Q-Max policy for controller {c+1} (eps={epsilon:.2f}) using {policy_output} returning controls {str(control_data)} based on {q_factors}",
                }
            )

        # log.debug(f"Q-Max policy with epsilon={epsilon} returning {control_data}")

        # Delete unused variables and force garbage collection
        del rl_df, q_factors

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
            self.config.control.reinforcement_learning,
            self.time_info.t,
            self.config.product,
            state_columns,
            action_size,
        )

        # Agent is now ready to perform control
        self.control_ready = True

        # Delete unused variables and force garbage collection
        del history, state_columns

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
        scaled_data = self.data_module.scale_dataframe_from_scaling_dict(data, self.uid)

        # TODO Add generalized check for this type of circumstance
        scaled_data = scaled_data.iloc[:-1]

        # Convert and push data to the replay buffers
        self.control_module.push_data_to_replay_buffer(
            uid=self.uid,
            data=data,
            rl_config=self.rl_config,
            state_columns=state_columns,
            control_columns=control_columns,
            reward_columns=reward_columns,
            simulation=simulation,
        )

        # Delete unused variables and force garbage collection
        del data, scaled_data, product, control_columns, reward_columns

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

    def assign_to_asset(self, asset: AssetType) -> None:
        self.asset = asset

    def assign_to_shadow_asset(self, shadow_asset: AssetType) -> None:
        self.shadow_asset = shadow_asset

    def save_config(self, filepath: str) -> None:
        # Dump agent configuration to JSON file
        with open(filepath, "w") as f:
            config_json = self.config.model_dump()
            json.dump(config_json, f, indent=4)

    def update_modules(
        self,
        data_module: DataModule,
        control_module: ControlModule,
        # prediction_module: PredictionModule,
    ) -> None:
        # Modules
        self.data_module = data_module
        self.control_module = control_module
        # self.prediction_module = prediction_module

        # Agent config and related quantities
        self.rl_config = self.config.control.reinforcement_learning

        log.debug(
            {
                LOG_SIM_T_KEY: None if self.time_info is None else self.time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.uid})",
                LOG_METHOD_KEY: "update_modules",
                LOG_MESSAGE_KEY: "Agent updated modules references.",
            }
        )

    def update_time_info(self, time_info: TimeInfo) -> None:
        self.time_info = time_info

    def update_weather_info(self, weather_info: WeatherInfo) -> None:
        self.weather_info = weather_info

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
