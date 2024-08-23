import numpy as np
import h5py
import pytest

from neuraflux.utils_hdf5 import (
    save_array_to_hdf5,
    append_to_dataset,
    retrieve_data_from_hdf5,
)


@pytest.fixture
def hdf5_file(tmp_path):
    """Fixture to create a temporary HDF5 file for testing."""
    file_path = tmp_path / "test_data.h5"
    yield str(file_path)
    # Cleanup: remove file after test (if it exists)
    file_path.unlink(missing_ok=True)


def test_save_array_to_hdf5(hdf5_file):
    # Create a 3D numpy array
    array = np.random.rand(10, 10, 10)

    # Save the array to HDF5
    save_array_to_hdf5(array, hdf5_file, "initial_states")

    # Check if the data was saved correctly
    with h5py.File(hdf5_file, "r") as f:
        assert "initial_states" in f
        data = np.array(f["initial_states"])
        np.testing.assert_array_equal(data, array)


def test_append_to_dataset(hdf5_file):
    # First save a 3D numpy array
    initial_array = np.random.rand(10, 10, 10)
    save_array_to_hdf5(initial_array, hdf5_file, "states")

    # Create new data to append
    new_data = np.random.rand(5, 10, 10)
    append_to_dataset(hdf5_file, "states", new_data)

    # Check if data was appended correctly
    with h5py.File(hdf5_file, "r") as f:
        data = np.array(f["states"])
        assert data.shape[0] == 15  # Original 10 + 5 new
        np.testing.assert_array_equal(data[:10, :, :], initial_array)
        np.testing.assert_array_equal(data[10:, :, :], new_data)


def test_retrieve_data_from_hdf5(hdf5_file):
    # Save a 3D numpy array
    array = np.random.rand(10, 10, 10)
    save_array_to_hdf5(array, hdf5_file, "retrieval_test")

    # Retrieve specific slices
    retrieved_data = retrieve_data_from_hdf5(
        hdf5_file, "retrieval_test", indices=slice(5, 8)
    )
    np.testing.assert_array_equal(retrieved_data, array[5:8, :, :])

    # Retrieve full array if no indices provided
    full_data = retrieve_data_from_hdf5(hdf5_file, "retrieval_test")
    np.testing.assert_array_equal(full_data, array)


# This test demonstrates the case where we attempt to append data to a non-existent dataset
def test_append_failure(hdf5_file):
    # Attempt to append data to a non-existent dataset should raise ValueError
    new_data = np.random.rand(5, 10, 10)
    with pytest.raises(ValueError):
        append_to_dataset(hdf5_file, "nonexistent_dataset", new_data)
