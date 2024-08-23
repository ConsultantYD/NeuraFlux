import h5py
import numpy as np
import logging as log


def save_array_to_hdf5(array: np.ndarray, file_path: str, dataset_name: str) -> None:
    """Save a NumPy array as a new dataset in an HDF5 file, ensuring that the dataset is resizable.

    Args:
    array (np.ndarray): The NumPy array to store.
    file_path (str): The file path of the HDF5 file.
    dataset_name (str): The name of the dataset within the HDF5 file.
    """
    try:
        with h5py.File(file_path, "a") as f:
            if dataset_name in f:
                log.error(f"Dataset '{dataset_name}' already exists.")
                raise ValueError(f"Dataset '{dataset_name}' already exists.")
            # Specify maxshape as None on resizeable dimensions, here assuming the first dimension
            f.create_dataset(
                dataset_name,
                data=array,
                compression="gzip",
                chunks=True,
                maxshape=(None, *array.shape[1:]),
            )
            log.debug(f"Array stored successfully in dataset '{dataset_name}'.")
    except Exception as e:
        log.error(f"Failed to store array: {e}")
        raise


def append_to_dataset(file_path: str, dataset_name: str, new_data: np.ndarray) -> None:
    """Append new data to an existing dataset in an HDF5 file along the first axis.

    Args:
    file_path (str): The file path of the HDF5 file.
    dataset_name (str): The name of the dataset to append to.
    new_data (np.ndarray): The new data to append.
    """
    try:
        with h5py.File(file_path, "a") as f:
            if dataset_name not in f:
                log.error(f"Dataset '{dataset_name}' does not exist.")
                raise ValueError(f"Dataset '{dataset_name}' does not exist.")
            dataset = f[dataset_name]
            dataset.resize((dataset.shape[0] + new_data.shape[0]), axis=0)
            dataset[-new_data.shape[0] :] = new_data
            log.debug(f"Data appended successfully to dataset '{dataset_name}'.")
    except Exception as e:
        log.error(f"Failed to append data: {e}")
        raise


def retrieve_data_from_hdf5(
    file_path: str, dataset_name: str, indices=None
) -> np.ndarray:
    """Retrieve data from an HDF5 dataset, optionally only specific indices.

    Args:
    file_path (str): The file path of the HDF5 file.
    dataset_name (str): The name of the dataset to retrieve data from.
    indices: Optional; the indices of the slices to load. If None, the whole dataset is returned.

    Returns:
    np.ndarray: The requested data.
    """
    try:
        with h5py.File(file_path, "r") as f:
            dataset = f[dataset_name]
            if indices is None:
                data = np.array(dataset[:])
            else:
                data = np.array(dataset[indices])
            log.debug(f"Data retrieved successfully from dataset '{dataset_name}'.")
            return data
    except Exception as e:
        log.error(f"Failed to retrieve data: {e}")
        raise
