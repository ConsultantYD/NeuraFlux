import json
import os
import shutil

import dill

from neuraflux.agency.dqn import DDQNPREstimator
from neuraflux.agency.replay_buffer import ReplayBuffer
from neuraflux.global_variables import METADATA_FILE_KEY, REPLAY_BUFFERS_FILENAME


def get_entities_in_registry(registry_dir: str) -> list[str]:
    """Get the entities in a registry, that is, the folders
    or files in the given directory.

    Args:
        registry_dir (str): The directory of the registry.

    Returns:
        list[str]: The entities in the registry.
    """
    # Return empty list if the directory does not exist
    if not os.path.exists(registry_dir):
        return []

    # Get all files in the directory
    directory_content = os.listdir(registry_dir)

    # Keep only directories
    entities_list = [
        f for f in directory_content if os.path.isdir(os.path.join(registry_dir, f))
    ]

    return entities_list


def push_replay_buffer_to_registry(
    registry_dir: str,
    name: str,
    replay_buffer: ReplayBuffer,
    metadata: dict[str, str | int | float | bool] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Push a replay buffer to the registry.

    Args:
        registry_dir (str): The directory of the registry.
        name (str): The name of the replay buffer.
        replay_buffer (ReplayBuffer): The replay buffer to push.
        metadata (dict[str, str | int | float | bool], optional): Metadata to save with the replay buffer. Defaults to None.
        overwrite (bool, optional): Whether to overwrite the existing entity. Defaults to False.
    """
    directory_path = os.path.join(registry_dir, name)

    # Delete previous entity if it exists and overwrite is True
    if overwrite and os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Save the replay buffer as a pickle file
    filepath = os.path.join(directory_path, REPLAY_BUFFERS_FILENAME)
    with open(filepath, "wb") as f:
        dill.dump(replay_buffer, f)

    # Save the metadata
    metadata = {} if metadata is None else metadata
    metadata_path = os.path.join(directory_path, METADATA_FILE_KEY)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def load_replay_buffer_from_registry(
    registry_dir: str,
    name: str,
) -> tuple[ReplayBuffer, dict[str, str | int | float | bool]]:
    """Load a replay buffer from the registry.

    Args:
        registry_dir (str): The directory of the registry.
        name (str): The name of the replay buffer.

    Returns:
        tuple[ReplayBuffer, dict[str, str | int | float | bool]]: The replay buffer and its metadata.
    """
    # Load the replay buffer
    directory_path = os.path.join(registry_dir, name)
    filepath = os.path.join(directory_path, REPLAY_BUFFERS_FILENAME)
    with open(filepath, "rb") as f:
        replay_buffer = dill.load(f)

    # Load the metadata
    metadata_path = os.path.join(directory_path, METADATA_FILE_KEY)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return replay_buffer, metadata


def push_dqn_estimator_to_registry(
    registry_dir: str,
    name: str,
    estimator: DDQNPREstimator,
    metadata: dict[str, str | int | float | bool] | None = None,
    overwrite: bool = False,
) -> None:
    """Push a DQN estimator to the registry.

    Args:
        registry_dir (str): The directory of the registry.
        name (str): The name of the estimator.
        estimator (DDQNPREstimator): The DQN estimator to push.
        metadata (dict[str, str | int | float | bool], optional): Metadata to save with the estimator. Defaults to None.
        overwrite (bool, optional): Whether to overwrite the existing entity. Defaults to False.
    """
    directory_path = os.path.join(registry_dir, name)

    # Delete previous entity if it exists and overwrite is True
    if overwrite and os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Save the estimator
    estimator.to_file(directory_path)

    # Save the metadata
    metadata = {} if metadata is None else metadata
    metadata_path = os.path.join(directory_path, METADATA_FILE_KEY)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def load_dqn_estimator_from_registry(
    registry_dir: str,
    name: str,
) -> tuple[DDQNPREstimator, dict[str, str | int | float | bool]]:
    """Load a DQN estimator from the registry.

    Args:
        registry_dir (str): The directory of the registry.
        name (str): The name of the estimator.

    Returns:
        tuple[DDQNPREstimator, dict[str, str | int | float | bool]]: The DQN estimator and its metadata.
    """
    # Load the estimator
    directory_path = os.path.join(registry_dir, name)
    estimator = DDQNPREstimator.from_file(directory_path)

    # Load the metadata
    metadata_path = os.path.join(directory_path, METADATA_FILE_KEY)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return estimator, metadata
