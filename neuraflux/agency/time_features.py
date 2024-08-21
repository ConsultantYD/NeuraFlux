import numpy as np
import pandas as pd


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
