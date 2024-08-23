import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest
from neuraflux.agency.data_module import DataModule
from neuraflux.global_variables import CONTROL_KEY, FILE_SCALING


@pytest.fixture
def data_module(tmp_path):
    # Create an instance of DataModule with a temporary directory
    base_dir = os.path.join(tmp_path, "agents_data")
    module = DataModule(base_dir=str(base_dir))
    return module


def test_initialize_new_agent_data_infrastructure(data_module):
    uid = "test_agent"
    signal_info_dict = {}  # Assuming you need a signal_info_dict for testing
    data_module.initialize_new_agent_data_infrastructure(uid, signal_info_dict)

    # Check if the folder and files are correctly created
    agent_dir = os.path.join(data_module.base_dir, uid)
    assert os.path.isdir(agent_dir), "Agent directory should exist"
    assert os.path.isfile(
        os.path.join(agent_dir, FILE_SCALING + ".json")
    ), "Scaling file should exist"


def test_store_and_get_trajectories(data_module):
    uid = "test_agent"
    data_module.initialize_new_agent_data_infrastructure(uid, {})

    index = dt.datetime.now()
    trajectory_data = pd.DataFrame(np.random.rand(10, 3), columns=["X", "Y", "Z"])
    data_module.store_trajectory(uid, index, trajectory_data)

    # Retrieve the trajectory
    retrieved_trajectories = data_module.get_trajectories(uid, index)
    assert len(retrieved_trajectories) == 1, "There should be one trajectory"
    pd.testing.assert_frame_equal(retrieved_trajectories[0], trajectory_data)


def test_store_data_in_table(data_module):
    uid = "test_agent"
    data_module.initialize_new_agent_data_infrastructure(uid, {})

    # Mock data to store
    table_name = "test_table"
    timestamp = dt.datetime.now()
    data = {"temperature": 22.5}
    data_columns = ["temperature"]

    data_module.store_data_in_table_at_time(
        uid, table_name, timestamp, data, data_columns
    )

    # Check if data is stored correctly
    retrieved_df = data_module.get_data_from_table(uid, table_name)
    assert len(retrieved_df) == 1, "Data should be successfully stored"
    assert (
        retrieved_df.iloc[0]["temperature"] == 22.5
    ), "Stored temperature should match the input"


def test_data_augmentation(data_module):
    uid = "test_agent"
    data_module.initialize_new_agent_data_infrastructure(uid, {})

    # Create a DataFrame with control data
    df = pd.DataFrame({CONTROL_KEY + "1": [1, 2, 3], CONTROL_KEY + "2": [4, 5, 6]})
    control_power_mapping = {1: 100, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}

    # Add 5mn datetime index
    df.index = pd.date_range("2023-01-01", periods=len(df), freq="5T")

    # Augment DataFrame with virtual metering data
    augmented_df = data_module.augment_dataframe_with_virtual_metering_data(
        df, control_power_mapping
    )
    assert "power" in augmented_df.columns, "Power column should be added"
    assert "energy" in augmented_df.columns, "Energy column should be added"

    # Augment DataFrame with tariff data
    tariff_str = "ONTARIO_GEN_TOU"
    augmented_df = data_module.augment_dataframe_with_tariff_data(
        augmented_df, tariff_str
    )
    print(augmented_df)
    assert (
        "tariff_$" in augmented_df.columns
    ), "Price column should be added after tariff augmentation"
