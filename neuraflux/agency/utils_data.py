import datetime as dt
import os

import numpy as np
import pandas as pd
from croniter import croniter

from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.tariffs import AvailableTariffsEnum
from neuraflux.global_variables import (
    CONTROL_KEY,
    ENERGY_KEY,
    POWER_KEY,
    PRICE_KEY,
    TIMESTAMP_KEY,
    TIMESTAMP_PARTITION_COL,
)
from neuraflux.local_typing import AssetType
from neuraflux.schemas.agency import AgentConfig, SignalTags


def get_columns_with_tag(agent_config: AgentConfig, tag: SignalTags) -> list[str]:
    """
    Return a list of column names that contain the given tag.
    Args:
        agent_config (AgentConfig): The agent configuration object.
        tag (SignalTags): The tag to filter by.
    Returns:
        list[str]: A list of column names that contain the given tag.
    """
    signal_infos = agent_config.data.signals_info
    return [k for k, v in signal_infos.items() if tag in v.tags]


def get_x_columns(agent_config: AgentConfig) -> list[str]:
    """
    Return a list of column names for the state, control, and exogenous variables.
    Args:
        agent_config (AgentConfig): The agent configuration object.
    Returns:
        list[str]: A list of column names for the state, control, and exogenous variables.
    """
    return get_columns_with_tag(agent_config, SignalTags.STATE)


def get_u_columns(agent_config: AgentConfig) -> list[str]:
    """
    Return a list of column names for the control variables.
    Args:
        agent_config (AgentConfig): The agent configuration object.
    Returns:
        list[str]: A list of column names for the control variables.
    """
    return get_columns_with_tag(agent_config, SignalTags.CONTROL)


def get_w_columns(agent_config: AgentConfig) -> list[str]:
    """
    Return a list of column names for the exogenous variables.
    Args:
        agent_config (AgentConfig): The agent configuration object.
    Returns:
        list[str]: A list of column names for the exogenous variables.
    """
    return get_columns_with_tag(agent_config, SignalTags.EXOGENOUS)


def get_active_config_based_on_duration(
    duration_s: float, config_dict: dict[int, object]
) -> object:
    """
    Get the active configuration based on the duration in seconds.

    Args:
        duration_s (float): The duration in seconds.
        config_dict (dict[int, object]): A dictionary with integer keys representing
                                         duration thresholds and values as configuration objects.
    Returns:
        object: The active configuration corresponding to the greatest threshold less than or equal to duration_s.
                Returns None if no threshold is met.
    """
    active_config = None
    # Sort the thresholds (keys) in ascending order
    for threshold in sorted(config_dict.keys()):
        if duration_s >= threshold:
            active_config = config_dict[threshold]
        else:
            break
    return active_config


def push_df_as_partitionned_parquet(
    df: pd.DataFrame,
    table_path: str,
    partition_cols: list[str] | None = None,
) -> None:
    """
    Push a DataFrame to parquet format, partitioned by the given columns.
    Args:
        df (pd.DataFrame): The DataFrame to save.
        directory (str): The directory where the parquet file will be saved.
        partition_cols (list[str]): The columns to use for partitioning. Will use
            the timestamp column by default, if present.
    """
    # Ensure the directory exists
    os.makedirs(table_path, exist_ok=True)

    # Create a new column with just the date for partitioning
    if partition_cols is None and TIMESTAMP_KEY in df.columns:
        partition_cols = [TIMESTAMP_PARTITION_COL]
        df[TIMESTAMP_PARTITION_COL] = df[TIMESTAMP_KEY].dt.date
    df.to_parquet(table_path, partition_cols=partition_cols, index=False)


def read_parquet_table(directory: str, order_by_timestamp: bool = True) -> pd.DataFrame:
    """
    Read a (set of) parquet file(s) from the given directory and return a DataFrame.
    Args:
        directory (str): The path to the parquet file.
        order_by_timestamp (bool): Whether to sort the DataFrame by timestamp.
    Returns:
        pd.DataFrame: The DataFrame containing the data from the parquet file.
    """
    df = pd.read_parquet(directory)
    # Order by timestamp, if available and desired
    if order_by_timestamp and TIMESTAMP_KEY in df.columns:
        # Sort by timestamp
        df = df.sort_values(by=[TIMESTAMP_KEY], ascending=True)
    # Drop partition columns, only used internally for storage
    if TIMESTAMP_PARTITION_COL in df.columns:
        df = df.drop(columns=[TIMESTAMP_PARTITION_COL])
    return df


def cron_matches(t: dt.datetime, cron_expr: str) -> bool:
    """
    Check if the given datetime matches the cron expression.
    Args:
        t (dt.datetime): The datetime to check.
        cron_expr (str): The cron expression to match against.
    Returns:
        bool: True if the datetime matches the cron expression, False otherwise.
    """
    # Cron expressions work with minute resolution.
    t = t.replace(second=0, microsecond=0)
    # Create a base time one minute before the target.
    base = t - dt.timedelta(minutes=1)
    itr = croniter(cron_expr, base)
    # The next scheduled time should equal the target time if it matches.
    return itr.get_next(dt.datetime) == t


def collect_signals_from_asset(
    asset: AssetType, tracked_signals: list[str]
) -> dict[str, float]:
    """
    Collect signals from the asset and store them in a dictionary.
    Args:
        asset (AssetType): The asset from which to collect signals.
        tracked_signals (list[str]): The list of signals to track.
    Returns:
        dict[str, float]: A dictionary containing the tracked signals.
    """
    signals_dict = {}
    for signal in tracked_signals:
        # Sample signals from the asset
        signal_value = asset.get_signal(signal)

        # Store array-like values with a suffix
        if isinstance(signal_value, (list, tuple, np.ndarray)):
            for i, value in enumerate(signal_value):
                signals_dict[signal + "_" + str(i + 1)] = value
        else:
            signals_dict[signal] = signal_value
    return signals_dict


def add_vm_data_to_df(
    df: pd.DataFrame,
    control_power_mapping: dict[int, float],
    timestep_mn: int = 5,
) -> pd.DataFrame:
    """
    Add virtual metering data to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to augment.
        control_power_mapping (dict[int, float]): The mapping of control
            values to power values.
        timestep_mn (int, optional): The time step in minutes. Defaults to 5.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """

    # Work with a copy of the DataFrame
    df = df.copy()

    # Identify control columns
    control_columns = [col for col in df.columns if col.startswith(CONTROL_KEY)]

    # Map each control column to its power and sum them
    df[POWER_KEY] = (
        df[control_columns].apply(lambda x: x.map(control_power_mapping)).sum(axis=1)
    )

    # Calculate energy from power, assuming 5mn time steps
    df[ENERGY_KEY] = df[POWER_KEY] * timestep_mn / 60

    del control_columns

    return df


def add_tariff_data_to_df(df: pd.DataFrame, tariff_str: str) -> pd.DataFrame:
    """Augment the dataframe with tariff information.

    Args:
        df (pd.DataFrame): The dataframe to augment.
        tariff_str (str): The name of the tariff to use.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """
    # Work with a copy of the dataframe
    df = df.copy()

    # Build Tariff object for tariff name
    tariff = AvailableTariffsEnum.from_string(tariff_str)

    # Calculate the price vector
    df = tariff.calculate_price_vector(df)

    # Delete unused variables
    del tariff

    return df


def add_product_data_to_df(df: pd.DataFrame, product_str: str) -> pd.DataFrame:
    """
    Augment the dataframe with tariff information.

    Args:
        df (pd.DataFrame): The dataframe to augment.
        product_str (str): The name of the product to use.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """
    # Work with a copy of the dataframe
    df = df.copy()

    # Build Tariff object for tariff name
    product = AvailableProductsEnum.from_string(product_str)

    # Calculate rewards
    reward_columns = product.get_reward_names()
    df[reward_columns] = product.calculate_rewards(df)

    # Calculate dones
    df = product.calculate_dones(df)

    # Add any custome features to the dataframe
    df = product.add_features(df)

    # Calculate the total price
    df[PRICE_KEY] = product.calculate_total_price(df)

    # Delete unused variables
    del product

    return df


def tf_all_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    """Add all cyclic time features to the input dataframe"""
    df = tf_cyclic_hour(df)
    df = tf_cyclic_day(df)
    df = tf_cyclic_weekday(df)
    df = tf_cyclic_month(df)
    df = tf_cyclic_year(df)
    return df


def tf_cyclic_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Incremental from 0 to 1 in 1 hour"""
    t = df.copy().index.minute  # type: ignore
    df["tf_cos_h"], df["tf_sin_h"] = cyclic_time_features(t, 60)
    return df


def tf_cyclic_day(df: pd.DataFrame) -> pd.DataFrame:
    """Incremental from 0 to 1 in 24 hours"""
    t = df.copy().index.hour + df.copy().index.minute / 60  # type: ignore
    df["tf_cos_d"], df["tf_sin_d"] = cyclic_time_features(t, 24)
    return df


def tf_1_hot_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 7 columns for each day of the week, and puts a 1 if it's that
    day.
    """
    day_indicator_columns = [
        "tf_mon",
        "tf_tue",
        "tf_wed",
        "tf_thu",
        "tf_fri",
        "tf_sat",
        "tf_sun",
    ]  # day of week indicators
    for i, ind_col in enumerate(day_indicator_columns):
        df[ind_col] = 0
        df.loc[df.index.dayofweek == i, ind_col] = 1  # type: ignore
    return df


def tf_cyclic_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Incremental from 0 to 1 in 7 days"""
    t = df.copy().index.weekday  # type: ignore
    df["tf_cos_w"], df["tf_sin_w"] = cyclic_time_features(t, 7)
    return df


def tf_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """Incremental from 0 to 1 in month"""
    t = df.copy().index.day  # type: ignore
    period = df.copy().index.daysinmonth  # type: ignore
    df["tf_cos_m"], df["tf_sin_m"] = cyclic_time_features(t, period)
    return df


def tf_cyclic_year(df: pd.DataFrame) -> pd.DataFrame:
    """Incremental from 0 to 1 over the whole year"""
    t = df.copy().index.dayofyear - 1  # type: ignore
    df["tf_cos_y"], df["tf_sin_y"] = cyclic_time_features(t, 365)
    return df


def cyclic_time_features(t, period: int):
    """Calculate the cyclic time features for a given period."""
    cos = np.cos(2 * t * np.pi / period)
    sin = np.sin(2 * t * np.pi / period)
    return (cos, sin)
