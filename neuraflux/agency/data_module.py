import datetime as dt
import json
import os

import h5py
import numpy as np
import pandas as pd

from neuraflux.agency.module import Module
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.scaling_utils import (
    load_scaler_info,
    save_scaler_info,
    scale_df_based_on_scaling_dict,
    update_scaling_dict_from_df,
    update_scaling_dict_from_signal_info,
)
from neuraflux.agency.tariffs import AvailableTariffsEnum
from neuraflux.agency.time_features import tf_all_cyclic
from neuraflux.global_variables import (
    CONTROL_KEY,
    ENERGY_KEY,
    FILE_SCALING,
    POWER_KEY,
    PRICE_KEY,
    TABLE_CONTROLS,
    TABLE_CONTROLS_SHADOW,
    TABLE_SIGNALS,
    TABLE_SIGNALS_SHADOW,
    TABLE_WEATHER,
)
from neuraflux.local_typing import (
    IndexType,
)
from neuraflux.schemas.agency import AgentConfig, SignalInfo, SignalTags


class DataModule(Module):
    """
    A class for managing data storage and retrieval during execution.
    """

    def __init__(self, base_dir: str = ""):
        super().__init__(base_dir)
        self.agent_signals_cache = {}
        self.agent_trajectory_columns_cache = {}

    def initialize_new_agent_data_infrastructure(
        self, uid: str, signal_info_dict: dict[str, SignalInfo]
    ) -> None:
        """
        Initialize the data infrastructure for a new agent.

        Parameters:
        -----------
        uid : str
            The unique identifier for the agent.
        signal_info_dict : dict of SignalInfo
            A dictionary containing details of signals of interest.
        """
        # Create agent directory
        agent_dir = os.path.join(self.base_dir, uid)
        if not os.path.isdir(agent_dir):
            os.makedirs(agent_dir)

        # Delete db file if it already exists
        db_file = os.path.join(agent_dir, f"{uid}.db")
        if os.path.isfile(db_file):
            os.remove(db_file)  # Remove if already exists

        # Initialize scaling dictionary and save as JSON
        with open(os.path.join(agent_dir, FILE_SCALING + ".json"), "w") as f:
            json.dump({}, f)
        self.update_scaler_info_from_signal_info(uid, signal_info_dict)

    def get_agent_data(
        self,
        uid: str,
        start_time: dt.datetime = None,
        end_time: dt.datetime = None,
        control_power_mapping: dict[int, float] | None = None,
        tariff: str | None = None,
        product: str | None = None,
        signals_data: bool = True,
        controls_data: bool = True,
        virtual_metering_data: bool = True,
        tariff_data: bool = True,
        product_data: bool = True,
        weather_data: bool = True,
        time_features: bool = False,
        shadow_asset: bool = False,
    ) -> pd.DataFrame:
        df = None

        # Signal data
        if signals_data:
            table = TABLE_SIGNALS_SHADOW if shadow_asset else TABLE_SIGNALS
            df = self.get_data_from_table(uid, table, start_time, end_time)

        # Control data
        if controls_data:
            table = TABLE_CONTROLS_SHADOW if shadow_asset else TABLE_CONTROLS
            df_controls = self.get_data_from_table(uid, table, start_time, end_time)
            if df is None:
                df = df_controls
            else:
                df = df.join(df_controls, how="outer")

        # Virtual Metering data
        if virtual_metering_data:
            df = self.augment_dataframe_with_virtual_metering_data(
                df, control_power_mapping
            )

        # Tariff data
        if tariff_data:
            df = self.augment_dataframe_with_tariff_data(df, tariff)

        # Weather data
        if weather_data:
            df_weather = self.get_data_from_table(
                uid, TABLE_WEATHER, start_time, end_time
            )
            if df is None:
                df = df_weather
            else:
                df = df.join(df_weather, how="outer")

        # Product data
        if product_data:
            df = self.augment_dataframe_with_product_data(df, product)

        # Time features
        if time_features:
            df = tf_all_cyclic(df)

        # Delete unused variables and force garbage collection
        del df_controls, df_weather

        return df

    def scale_dataframe_from_scaling_dict(
        self, df: pd.DataFrame, uid: str
    ) -> pd.DataFrame:
        # Work with a copy of the DataFrame
        df = df.copy()

        # Define scaler directory
        scaler_dir = os.path.join(self.base_dir, uid)

        # Load scaling dictionary
        scaling_dict = load_scaler_info(scaler_dir)

        # Scale the dataframe
        scaled_df = scale_df_based_on_scaling_dict(df, scaling_dict)

        # Delete unused variables and force garbage collection
        del df, scaling_dict

        return scaled_df

    def update_scaler_info_from_signal_info(
        self, uid: str, signal_info_dict: dict[str, SignalInfo]
    ):
        # Define scaler directory
        scaler_dir = os.path.join(self.base_dir, uid)

        # Load scaling dictionary
        scaling_dict = load_scaler_info(scaler_dir)

        # Update scaling dictionary
        scaling_dict = update_scaling_dict_from_signal_info(
            scaling_dict, signal_info_dict
        )

        # Save scaling dictionary
        save_scaler_info(scaler_dir, scaling_dict)

        # Delete unused variables and force garbage collection
        del signal_info_dict, scaling_dict

    def update_scaler_info_from_df(self, uid: str, df: pd.DataFrame):
        # Define scaler directory
        scaler_dir = os.path.join(self.base_dir, uid)

        # Load scaling dictionary
        scaling_dict = load_scaler_info(scaler_dir)

        # Update scaling dictionary
        scaling_dict = update_scaling_dict_from_df(df, scaling_dict, list(df.columns))

        # Save scaling dictionary
        save_scaler_info(scaler_dir, scaling_dict)

        # Delete unused variables and force garbage collection
        del df, scaling_dict

    def get_columns_with_tag(
        self, agent_config: AgentConfig, tag: SignalTags
    ) -> list[str]:
        """Return a list of column names that contain the given tag."""
        signal_infos = agent_config.data.signals_info
        return [k for k, v in signal_infos.items() if tag in v.tags]

    def store_trajectory(self, uid: str, index: IndexType, trajectory: pd.DataFrame):
        """Store or append a trajectory DataFrame in an HDF5 file under a single dataset."""

        # Verify if trajectory columns have been stored/cached
        if uid not in self.agent_trajectory_columns_cache:
            # Cache columns
            self.agent_trajectory_columns_cache[uid] = trajectory.columns
            # Store columns in a JSON file for long-term reference
            columns_file = os.path.join(self.base_dir, uid, "trajectory_columns.json")
            with open(columns_file, "w") as f:
                json.dump(trajectory.columns.tolist(), f)

        # Sanitize and ensure the trajectory DataFrame has the correct columns
        clean_trajectory = trajectory.copy()[self.agent_trajectory_columns_cache[uid]]

        # Store the trajectory data in an HDF5 file as Numpy arrays
        index_str = (
            str(index)
            if not isinstance(index, dt.datetime)
            else index.strftime("%Y%m%d%H%M%S")
        )
        hdf5_file = os.path.join(self.base_dir, uid, f"{uid}_trajectories.hdf5")
        dataset_name = f"{uid}/{index_str}"

        with h5py.File(hdf5_file, "a") as f:
            data_array = clean_trajectory.to_numpy()[
                np.newaxis, ...
            ]  # Add new axis to make it 3D

            if dataset_name in f:
                dataset = f[dataset_name]
                dataset.resize((dataset.shape[0] + 1,) + dataset.shape[1:])
                dataset[-1, :, :] = data_array
            else:
                maxshape = (None,) + data_array.shape[
                    1:
                ]  # Allow unlimited appending along the first axis
                f.create_dataset(
                    dataset_name,
                    data=data_array,
                    maxshape=maxshape,
                    compression="gzip",
                    chunks=True,
                )

        # Delete unused variables and force garbage collection
        del clean_trajectory, data_array, trajectory

    def get_trajectories(self, uid: str, index: IndexType):
        """Retrieve all trajectory DataFrames for a given uid and index from an HDF5 file."""
        index_str = (
            str(index)
            if not isinstance(index, dt.datetime)
            else index.strftime("%Y%m%d%H%M%S")
        )
        hdf5_file = os.path.join(self.base_dir, uid, f"{uid}_trajectories.hdf5")
        dataset_name = f"{uid}/{index_str}"

        trajectory_cols = self.agent_trajectory_columns_cache[uid]

        with h5py.File(hdf5_file, "r") as f:
            if dataset_name in f:
                dataset = f[dataset_name]
                trajectories = [
                    pd.DataFrame(data, columns=trajectory_cols) for data in dataset
                ]

                # Delete unused variables
                del dataset, trajectory_cols

                return trajectories
            else:
                return []

    def augment_dataframe_with_virtual_metering_data(
        self,
        df: pd.DataFrame,
        control_power_mapping: dict[int, float],
        timestep_mn: int = 5,
    ) -> pd.DataFrame:
        """Calculate the virtual metering data from asset signal data."""

        # Work with a copy of the DataFrame
        df = df.copy()

        # Identify control columns
        control_columns = [col for col in df.columns if col.startswith(CONTROL_KEY)]

        # Map each control column to its power and sum them
        df[POWER_KEY] = (
            df[control_columns]
            .apply(lambda x: x.map(control_power_mapping))
            .sum(axis=1)
        )

        # Calculate energy from power, assuming 5mn time steps
        df[ENERGY_KEY] = df[POWER_KEY] * timestep_mn / 60

        return df

    def augment_dataframe_with_tariff_data(
        self, df: pd.DataFrame, tariff_str: str
    ) -> pd.DataFrame:
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

    def augment_dataframe_with_product_data(
        self, df: pd.DataFrame, product_str: str
    ) -> pd.DataFrame:
        """Augment the dataframe with tariff information.

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
