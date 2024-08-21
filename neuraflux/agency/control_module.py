import datetime as dt
import json
import logging as log
import os
from typing import Any

import dill
import numpy as np
import pandas as pd

from neuraflux.agency.control_utils import (
    convert_data_to_experience,
    convert_data_to_state,
    get_full_state_signals_from_rl_config,
)
from neuraflux.agency.ddqn import DDQNPREstimator
from neuraflux.agency.module import Module
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.replay_buffer import ReplayBuffer
from neuraflux.agency.rl_training import (
    simple_training_loop,
)
from neuraflux.global_variables import (
    FILE_REAL_REPLAY_BUFFER,
    FILE_SIM_REPLAY_BUFFER,
    LOG_SIM_T_KEY,
    LOG_ENTITY_KEY,
    LOG_METHOD_KEY,
    LOG_MESSAGE_KEY,
    TABLE_DQN_TRAINING,
)
from neuraflux.schemas.agency import RLConfig
from neuraflux.time_ref import convert_datetime_to_unix
from neuraflux.utils_sql import (
    add_dataframe_to_table,
)


class ControlModule(Module):
    """
    A class for managing reinforcement learning and control-related operations.
    """

    def rl_training(
        self,
        uid: str,
        rl_config: RLConfig,
        index: dt.datetime,
        product_str: str,
        state_columns: list[str],
        action_size: int,
    ) -> None:
        # Log the start of the training process
        log.debug(
            {
                LOG_SIM_T_KEY: index,
                LOG_ENTITY_KEY: "Control Module",
                LOG_METHOD_KEY: "rl_training",
                LOG_MESSAGE_KEY: f"Initializing training with product {product_str}, state columns {state_columns}, action size {action_size} and RLConfig {str(rl_config)}",
            }
        )

        # Retrieve replay buffers
        real_buffer = self.get_replay_buffer(uid, rl_config=rl_config, simulation=False)

        # Obtain production Q estimator, using latest model
        available_models = self.get_available_models_in_registry(uid)
        if len(available_models) == 0:
            product = AvailableProductsEnum.from_string(product_str)
            n_rewards = len(product.get_reward_names())
            del product  # Delete unused variable
            q_estimator = self.initialize_agent_q_estimator_from_config(
                rl_config, n_rewards, state_columns, action_size
            )
        else:
            q_estimator = self.get_model_from_registry(uid, available_models[-1])

        model_name = index.strftime("%y_%m_%d__%H_%M")
        for target_iterator in range(rl_config.n_target_updates):
            q_estimator, real_buffer, err_list, times_dict = simple_training_loop(
                replay_buffer_real=real_buffer,
                q_estimator=q_estimator,
                sampling_size=rl_config.experience_sampling_size,
                n_sampling_iters=rl_config.n_sampling_iters,
                learning_rate=rl_config.learning_rate,
                n_fit_epochs=rl_config.n_fit_epochs,
                tf_batch_size=rl_config.tf_batch_size,
            )
            q_estimator.update_target_model()

            # Store training details in table
            training_df = pd.DataFrame(
                {
                    "td_rmse": err_list,
                    "training_duration": times_dict["training"],
                    "batch_sampling_duration": times_dict["batch_sampling"],
                    "global_td_errors_calculation_duration": times_dict[
                        "global_td_errors_calculation"
                    ],
                    "td_error_PER_update_duration": times_dict["td_error_per_update"],
                }
            )
            training_df["training_timestamp"] = convert_datetime_to_unix(index)
            training_df["model_name"] = model_name
            training_df["target_iterator"] = target_iterator
            training_df["experience_sampling_size"] = rl_config.experience_sampling_size
            training_df["n_sampling_iters"] = rl_config.n_sampling_iters
            training_df["learning_rate"] = rl_config.learning_rate
            training_df["n_fit_epochs"] = rl_config.n_fit_epochs
            training_df["tf_batch_size"] = rl_config.tf_batch_size
            training_df["uid"] = uid
            db_connection = self.create_connection_to_agent_db(uid)
            add_dataframe_to_table(
                training_df,
                db_connection,
                TABLE_DQN_TRAINING,
                index_col=None,
                use_index=False,
            )

        # Convert time index to model name str
        self.push_model_to_registry(uid, model_name, q_estimator)
        self.save_replay_buffer(uid, real_buffer, simulation=False)

        # Update training summary
        training_summary = self.get_training_summary(uid)
        if "n_trainings" not in training_summary:
            training_summary["n_trainings"] = 0
        training_summary["n_trainings"] += 1
        self.save_training_summary(uid, training_summary)

        # Log the start of the training process
        log.debug(
            {
                LOG_SIM_T_KEY: index,
                LOG_ENTITY_KEY: "Control Module",
                LOG_METHOD_KEY: "rl_training",
                LOG_MESSAGE_KEY: "Training completed successfully.",
            }
        )

        # Delete unused variables and force garbage collection
        del (
            real_buffer,
            q_estimator,
            available_models,
            err_list,
            training_df,
            db_connection,
        )

    def get_rl_training_data_table(self, uid: str) -> pd.DataFrame:
        # db_connection = self.create_connection_to_agent_db(uid)
        training_df = self.get_data_from_table(
            uid,
            TABLE_DQN_TRAINING,
        )
        return training_df

    def get_training_summary(self, uid: str) -> dict[str, Any]:
        filepath = os.path.join(self.base_dir, uid, "training_summary.json")
        training_summary = {}
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                training_summary = json.load(f)
        return training_summary

    def save_training_summary(self, uid: str, training_summary: dict[str, Any]) -> None:
        filepath = os.path.join(self.base_dir, uid, "training_summary.json")
        with open(filepath, "w") as f:
            json.dump(training_summary, f)

    # -----------------------------------------------------------------------
    # Q-FACTORS
    # -----------------------------------------------------------------------
    def get_raw_q_factors(
        self,
        uid: str,
        scaled_data: pd.DataFrame,
        state_columns: list[str],
        rl_config: RLConfig,
        model_name: None | str = None,
    ) -> None:
        # Work with a copy of the input data
        scaled_df = scaled_data.copy()

        # Convert the dataframe to its NumPy representation
        state_columns = get_full_state_signals_from_rl_config(rl_config, state_columns)
        seq_len = rl_config.history_length
        states = np.array(convert_data_to_state(scaled_df, state_columns, seq_len))

        # Obtain production Q estimator, using latest model
        available_models = self.get_available_models_in_registry(uid)
        model_name = available_models[-1] if model_name is None else model_name
        q_estimator = self.get_model_from_registry(uid, model_name)
        q_factors = q_estimator.forward_pass(states)

        # Delete unused variables and force garbage collection
        del scaled_df, states, q_estimator

        return q_factors

    def augment_df_with_q_factors(
        self,
        uid: str,
        data: pd.DataFrame,
        scaled_data: pd.DataFrame,
        state_columns: list[str],
        rl_config: RLConfig,
        model_name: None | str = None,
    ) -> None:
        # Work with a copy of the input datasets
        df = data.copy()
        scaled_data = scaled_data.copy()

        seq_len = rl_config.history_length
        q_factors = self.get_raw_q_factors(
            uid, scaled_data, state_columns, rl_config, model_name
        )

        # Add Q-factors to the dataframe and return
        # NOTE: q_factors are a list (n_controllers len) with elements of
        # dimension (seq_len, n_actions, n_rewards)
        for c, q_vals in enumerate(q_factors):
            for r in range(q_vals.shape[1]):
                q_cols = [f"Q{r+1}_C{c+1}_U{i+1}" for i in range(q_vals.shape[2])]
                df.loc[df.index.values[seq_len - 1 :], q_cols] = q_vals[:, r, :]

        # Delete unused variables and force garbage collection
        del data, scaled_data, q_factors, rl_config

        return df

    # -----------------------------------------------------------------------
    # REPLAY BUFFERS
    # -----------------------------------------------------------------------
    def get_replay_buffer(
        self, uid: str, rl_config: RLConfig, simulation: bool = False
    ) -> ReplayBuffer:
        FILENAME = FILE_SIM_REPLAY_BUFFER if simulation else FILE_REAL_REPLAY_BUFFER
        replay_buffer_file = os.path.join(self.base_dir, uid, FILENAME)
        if os.path.exists(replay_buffer_file):
            with open(replay_buffer_file, "rb") as f:
                replay_buffer = dill.load(f)
        else:
            replay_buffer = ReplayBuffer(
                max_len=rl_config.replay_buffer_size,
                prioritized_replay_alpha=rl_config.prioritized_replay_alpha,
            )
        return replay_buffer

    def save_replay_buffer(
        self, uid: str, replay_buffer: ReplayBuffer, simulation: bool = False
    ) -> None:
        FILENAME = FILE_SIM_REPLAY_BUFFER if simulation else FILE_REAL_REPLAY_BUFFER
        replay_buffer_file = os.path.join(self.base_dir, uid, FILENAME)
        with open(replay_buffer_file, "wb") as f:
            dill.dump(replay_buffer, f)

        # Delete unused variables and force garbage collection
        del replay_buffer

    def push_data_to_replay_buffer(
        self,
        uid: str,
        data: pd.DataFrame,
        rl_config: RLConfig,
        state_columns: list[str],
        control_columns: list[str],
        reward_columns: list[str],
        simulation: bool = False,
    ) -> None:
        # Get the corresponding agent's replay buffer
        replay_buffer = self.get_replay_buffer(uid, rl_config, simulation)

        # Convert the dataframe to its NumPy representation
        state_columns = get_full_state_signals_from_rl_config(rl_config, state_columns)
        seq_len = rl_config.history_length
        experience_batch = convert_data_to_experience(
            data, seq_len, state_columns, control_columns, reward_columns
        )

        # Log an example of the conversion to confirm proper operation
        log.debug(
            {
                LOG_SIM_T_KEY: None,
                LOG_ENTITY_KEY: "Control Module",
                LOG_METHOD_KEY: "push_data_to_replay_buffer",
                LOG_MESSAGE_KEY: f"Input data example: {data.iloc[0:seq_len+1]} \n"
                f"Exp s: {experience_batch[0][0]} \n"
                f"Exp a: {experience_batch[1][0]} \n"
                f"Exp r: {experience_batch[2][0]} \n"
                f"Exp ns: {experience_batch[3][0]} \n"
                f"Exp d: {experience_batch[4][0]} \n",
            }
        )

        # Add the experience samples from the dataframe to the replay buffer
        for experience in zip(*experience_batch):
            replay_buffer.add_experience_sample(experience=experience)

        # Save the updated replay buffer to file
        self.save_replay_buffer(uid, replay_buffer, simulation)

        # Delete unused variables and force garbage collection
        del data, experience_batch, replay_buffer

    # -----------------------------------------------------------------------
    # Q ESTIMATORS & REGISTRY
    # -----------------------------------------------------------------------
    def initialize_agent_q_estimator_from_config(
        self,
        rl_config: RLConfig,
        n_rewards: int,
        state_columns: list[str],
        action_size: int,
    ):
        # Define important variables
        state_columns = get_full_state_signals_from_rl_config(rl_config, state_columns)
        state_size = len(state_columns)
        action_size = (
            action_size if rl_config.action_size is None else rl_config.action_size
        )

        # Initialize Q estimator
        q_estimator = DDQNPREstimator(
            state_size=state_size,
            action_size=action_size,
            sequence_len=rl_config.history_length,
            n_rewards=n_rewards,
            n_controllers=rl_config.n_controllers,
            learning_rate=rl_config.learning_rate,
            discount_factor=rl_config.discount_factor,
        )

        # Delete unused variables and force garbage collection
        del rl_config, state_columns

        return q_estimator

    def get_available_models_in_registry(self, uid: str) -> list[str]:
        registry_dir = os.path.join(self.base_dir, uid, "ddqn_models")
        if not os.path.isdir(registry_dir):
            return []
        available_models = os.listdir(registry_dir)
        available_models.sort()  # Sort alphabetically
        return available_models

    def push_model_to_registry(
        self, uid: str, model_name: str, q_estimator: DDQNPREstimator
    ) -> None:
        model_dir = os.path.join(self.base_dir, uid, "ddqn_models")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        model_file = os.path.join(model_dir, model_name)
        q_estimator.to_file(model_file)

        # Delete unused variables and force garbage collection
        del q_estimator

    def get_model_from_registry(self, uid: str, model_name: str) -> DDQNPREstimator:
        model_file = os.path.join(self.base_dir, uid, "ddqn_models", model_name)
        model = DDQNPREstimator.from_file(model_file)
        return model

    def get_training_data(self, uid: str) -> pd.DataFrame:
        db_connection = self.create_connection_to_agent_db(uid)
        training_df = self.get_data_from_table(
            db_connection,
            TABLE_DQN_TRAINING,
        )
        # training_df = pd.read_sql(f"SELECT * FROM {TABLE_DQN_TRAINING}", db_connection)
        return training_df
