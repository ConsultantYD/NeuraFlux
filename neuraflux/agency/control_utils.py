from copy import copy

import numpy as np
import pandas as pd
import tensorflow as tf

from neuraflux.global_variables import DONE_KEY
from neuraflux.schemas.agency import RLConfig


def softmax(q_values, tau=1.0):
    """
    Computes the softmax of the given Q-values.
    """
    q_values_adj = q_values - np.max(q_values)  # for numerical stability
    exp_values = np.exp(q_values_adj / tau)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


def convert_data_to_state(
    data: pd.DataFrame, state_columns: list[str], seq_len: int
) -> np.ndarray:
    """Converts a dataframe of data from an asset observation
    dataframe to numpy states.
    """
    data = data.copy()
    states = data[state_columns]
    states_len_before = states.shape[0]
    old_states = states
    states = states.dropna()
    if states.shape[0] != states_len_before:
        print("Rows were removed !")
        print("before")
        print(old_states)
        print("after")
        print(states)

    states = states.values
    states = np.asarray(states).astype("float32")
    states_df_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        states, None, seq_len, batch_size=None
    )

    # Convert the dataset to a list of Numpy states
    # NOTE: ending the iterator will trigger TF warning
    states_list = []
    states_df_dataset_iterator = states_df_dataset.as_numpy_iterator()
    for _ in range(len(states_df_dataset)):
        gen_output = next(states_df_dataset_iterator, None)
        if gen_output is not None:
            states_list.append(gen_output)

    # TODO Try to convert back to ndarray
    return states_list


def convert_data_to_experience(
    data: pd.DataFrame,
    seq_len: int,
    state_columns: list[str],
    control_columns: list[str],
    reward_columns: list[str],
) -> tuple:
    """Converts a dataframe of data from an asset observation
    dataframe to a zip of experience tuples.

    Args:
        uid (Union[int, str]): The unique identifier of the asset.
        data (pd.DataFrame): The data to push to the replay buffer.
    """
    data = data.copy()
    # Convert the dataframe to its NumPy representations
    states = convert_data_to_state(data.iloc[:-1], state_columns, seq_len)
    # NOTE: Here we use int and not Int64, as we want an error if there is NA
    actions = data[control_columns].values[seq_len - 1 : -1].astype(int)
    rewards = data[reward_columns].values[seq_len - 1 : -1].astype(float)
    next_states = convert_data_to_state(data.iloc[1:], state_columns, seq_len)
    dones = (
        data[[DONE_KEY]]
        .values[seq_len - 1 : -1]
        .reshape(
            -1,
        )
    )
    td_errors = np.array([None for _ in range(len(states))])

    # Verify the presence of NaN or NA in any of the data
    if any(
        [
            np.isnan(states).any(),
            np.isnan(actions).any(),
            np.isnan(rewards).any(),
            np.isnan(next_states).any(),
            np.isnan(dones).any(),
        ]
    ):
        raise ValueError("NaN or NA values found in the data.")

    # Add the experience samples from the dataframe to the replay buffer
    experience = (states, actions, rewards, next_states, dones, td_errors)

    return experience


def get_full_state_signals_from_rl_config(
    rl_config: RLConfig, state_columns: list[str]
) -> list[str]:
    """Augment the state signals with time features and other options
    if user selected so.

    Args:
        rl_config (RLConfig): Agent's control reinforcement learning config.
        state_columns (list[str]): List of state signals.
    Returns:
        list[str]: List of augmented state signals.
    """

    if rl_config.state_signals is not None:
        state_columns = rl_config.state_signals
    state_signals = copy(state_columns)

    # Extract boolean time feature options
    tf_hourly = rl_config.add_hourly_time_features_to_state
    tf_daily = rl_config.add_daily_time_features_to_state
    tf_weekly = rl_config.add_weekly_time_features_to_state
    tf_monthly = rl_config.add_monthly_time_features_to_state

    # Add time features to state signals if necessary
    if tf_hourly:
        state_signals += ["tf_cos_h", "tf_sin_h"]
    if tf_daily:
        state_signals += ["tf_cos_d", "tf_sin_d"]
    if tf_weekly:
        state_signals += ["tf_cos_w", "tf_sin_w"]
    if tf_monthly:
        state_signals += ["tf_cos_m", "tf_sin_m"]

    return state_signals
