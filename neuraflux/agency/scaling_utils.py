import json
import os
from enum import Enum, unique

import numpy as np
import pandas as pd

from neuraflux.global_variables import FILE_SCALING
from neuraflux.schemas.agency import ScalingMetadata, SignalInfo


def load_scaler_info(directory: str) -> dict[str, ScalingMetadata]:
    scaling_file = os.path.join(directory, FILE_SCALING + ".json")
    with open(scaling_file, "r") as f:
        scaling_dict = json.load(f)
    for signal, info in scaling_dict.items():
        scaling_dict[signal] = ScalingMetadata(**info)
    return scaling_dict


def save_scaler_info(directory: str, scaling_dict: dict[str, ScalingMetadata]):
    scaling_file = os.path.join(directory, FILE_SCALING + ".json")
    scaling_dict_serializable = {k: v.model_dump() for k, v in scaling_dict.items()}
    with open(scaling_file, "w") as f:
        json.dump(scaling_dict_serializable, f, indent=4)


def update_scaling_dict_from_signal_info(
    scaling_dict: dict[str, ScalingMetadata],
    signal_info_dict: dict[str, SignalInfo],
):
    # Loop over all signals in the signal info dictionary
    for signal, info in signal_info_dict.items():
        if signal in scaling_dict:
            scaling_metadata_dict = scaling_dict[signal].model_dump()
        else:
            scaling_metadata_dict = {}

        # Update scaling metadata with signal info
        if info.min_value is not None:
            scaling_metadata_dict["min_value"] = info.min_value
        if info.max_value is not None:
            scaling_metadata_dict["max_value"] = info.max_value

        # Update wheter the signal is scalable or not
        scaling_metadata_dict["scalable"] = info.scalable

        # Update the entry
        scaling_dict[signal] = ScalingMetadata(**scaling_metadata_dict)

    return scaling_dict


def update_scaling_dict_from_df(
    df: pd.DataFrame,
    scaling_dict: dict[str, ScalingMetadata],
    columns_to_update: list[str] = [],
) -> dict[str, ScalingMetadata]:
    """
    Update the scaling dictionary based on the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to update the scaling dictionary with
    scaling_dict : dict[str, ScalingMetadata]
        The scaling dictionary to update
    columns_to_update : list[str]
        The columns to update in the scaling dictionary
    """
    df = df.copy()

    # Define list of columns to iterate over
    updated_columns = np.intersect1d(columns_to_update, df.columns)
    for col in updated_columns:
        col_min = df[col].min(skipna=True)
        col_max = df[col].max(skipna=True)

        if col in scaling_dict:
            scaling_metadata_dict = scaling_dict[col].model_dump()
        else:
            scaling_metadata_dict = {}

        # Update sampled signal info, updating values if necessary
        if (
            "min_sampled_value" in scaling_metadata_dict
            and scaling_metadata_dict["min_sampled_value"] is not None
        ):
            scaling_metadata_dict["min_sampled_value"] = min(
                col_min, scaling_metadata_dict["min_sampled_value"]
            )
        else:
            scaling_metadata_dict["min_sampled_value"] = col_min
        if (
            "max_sampled_value" in scaling_metadata_dict
            and scaling_metadata_dict["max_sampled_value"] is not None
        ):
            scaling_metadata_dict["max_sampled_value"] = max(
                col_max, scaling_metadata_dict["max_sampled_value"]
            )
        else:
            scaling_metadata_dict["max_sampled_value"] = col_max

        # Update the entry
        scaling_dict[col] = ScalingMetadata(**scaling_metadata_dict)

    return scaling_dict


@unique
class EnumScalingTypes(Enum):
    MAX = "max"
    MIN_MAX = "min_max"
    MINUS1_1 = "minus1_1"


def scale_df_based_on_scaling_dict(
    df: pd.DataFrame,
    scaling_dict: dict[str, ScalingMetadata],
    scaling_type: EnumScalingTypes = EnumScalingTypes.MAX,
) -> pd.DataFrame:
    """
    Scale the dataframe based on the scaling dictionary

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to scale
    scaling_dict : dict[str, ScalingMetadata]
        The scaling dictionary
    scaling_type : ScalingTypesEnum
        The scaling type to use
    """
    scaled_df = df.copy()

    for col in df.columns:
        if col in scaling_dict and scaling_dict[col].scalable:
            if (
                scaling_dict[col].min_value is not None
                and scaling_dict[col].max_value is not None
            ):
                min_val = scaling_dict[col].min_value
                max_val = scaling_dict[col].max_value
            elif (
                scaling_dict[col].min_sampled_value is not None
                and scaling_dict[col].max_sampled_value is not None
            ):
                min_val = scaling_dict[col].min_sampled_value
                max_val = scaling_dict[col].max_sampled_value
            else:
                raise ValueError(
                    f"Missing min and max values in scaling dictionary for {col}"
                )

            # Check for the case where min and max are the same
            if min_val == max_val:
                scaled_df[col] = 0  # or any other default value
            else:
                # Max scaling (default)
                if scaling_type == EnumScalingTypes.MAX:
                    x_scaled = df[col] / max_val

                # Min-Max scaling
                elif scaling_type == EnumScalingTypes.MIN_MAX:
                    x_scaled = (df[col] - min_val) / (max_val - min_val)

                # -1 to 1 scaling
                elif scaling_type == EnumScalingTypes.MINUS1_1:
                    x_scaled = 2 * x_scaled - 1

                scaled_df[col] = x_scaled

    return scaled_df
