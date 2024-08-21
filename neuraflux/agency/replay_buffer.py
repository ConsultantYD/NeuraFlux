from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, max_len: int = 20000, prioritized_replay_alpha: float = 0.5):
        self.max_len = max_len
        self.prioritized_replay_alpha = prioritized_replay_alpha

        self.storage = deque(maxlen=max_len)

    def __len__(self) -> int:
        return len(self.storage)

    def add_experience_sample(
        self,
        experience: tuple[
            np.ndarray[float], int, list[float], np.ndarray[float], bool, float
        ],
        def_err: float = 9999,
    ) -> None:
        """Adds a new experience sample to the replay buffer.

        Args:
            experience (Tuple[ np.ndarray[float], int, float, np.ndarray[float], bool, float ]): The experience sample to add to the replay buffer, where the first element is the state, the second element is the action, the third element is the reward, the fourth element is the next state, the fifth element is a boolean indicating whether the episode is done, and the sixth element is the error.
        """
        # Ensure error is numerical - update None error values
        if experience[5] is None:
            all_errors = self.get_experience_errors()
            max_err = def_err  # Use default error if no errors in buffer
            if len(all_errors > 0):
                max_err = np.max(all_errors)  # Else, max in buffer
            experience = list(experience)
            experience[5] = max_err
            experience = tuple(experience)

        # Store the experience sample
        self.storage.append(experience)

    def update_experience_at_index(self, index, experience):
        self.storage[index] = experience

    def get_experience_errors(self) -> np.ndarray[float]:
        """Returns the errors of the experience samples in the replay buffer.

        Returns:
            np.ndarray[float]: The errors of the experience samples in the replay buffer.
        """
        return np.array([exp[5] for exp in self.storage])

    def update_td_errors(
        self, td_errors: np.ndarray, indices: np.ndarray | None = None
    ) -> None:
        """Updates the TD errors of the specified experiences.

        Args:
            indices (np.ndarray): The indices of the experiences to update.
            td_errors (np.ndarray): The new TD errors.
        """
        # If just the TD errors were provided, update all experiences
        if indices is None:
            if not len(td_errors) == len(self.storage):
                raise ValueError(
                    "The number of TD errors must match the number of experiences in the replay buffer."
                )
            indices = np.arange(len(self.storage))

        # Loop over new errors and indices
        for idx, error in zip(indices, td_errors):
            experience = list(self.storage[idx])
            experience[5] = error  # Update the 6th field with the new TD error
            self.storage[idx] = tuple(experience)

    def get_prioritized_experience_samples(self, sampling_size: int | None):
        # Return all samples if batch size is None
        if sampling_size is None:
            return self.get_experience_samples()

        # Sanitize batch size to always be at least the size of storage
        sampling_size = min(sampling_size, len(self.storage))

        # Otherwise, compute priorities and sample based on them
        priorities = self.get_experience_priorities()
        indexes = self.get_indexes_based_on_priorities(sampling_size, priorities)
        experience_samples = self._get_experience_samples_from_indexes(indexes)
        sample_priorities = priorities[indexes]
        return experience_samples, sample_priorities, indexes

    def get_experience_samples(
        self, sampling_size: int | None = None
    ) -> tuple[
        np.ndarray[
            tuple[np.ndarray[float], int, float, np.ndarray[float], bool, float]
        ],
        np.ndarray[int],
    ]:
        # Retrieve all samples if batch size is None
        sampling_size = len(self.storage) if sampling_size is None else sampling_size

        # Sanitize batch size to always be at least the size of storage
        sampling_size = min(sampling_size, len(self.storage))

        # Randomly sample from the replay buffer
        indexes = np.random.choice(len(self.storage), size=sampling_size, replace=False)
        experience_samples = self._get_experience_samples_from_indexes(indexes)
        priorities = self.get_experience_priorities()[indexes]
        return experience_samples, priorities, indexes

    def get_indexes_based_on_priorities(
        self, sampling_size: int, priorities: np.ndarray[float] | None = None
    ) -> np.ndarray[int]:
        # Uniform provabilities if priorities are None
        priorities = (
            np.ones(len(self.storage)) / len(self.storage)
            if priorities is None
            else priorities
        )

        # Define indexes based on priority
        indexes = np.random.choice(
            len(self.storage), size=sampling_size, replace=False, p=priorities
        )

        return indexes

    def get_experience_priorities(self, epsilon: float = 1e-3) -> np.ndarray[float]:
        """Returns the normalized priorities of the experience samples in the replay buffer based on the error.

        Args:
            epsilon (float): Small value to ensure non-zero priority.

        Returns:
            np.ndarray[float]: The priorities of the experience samples in the replay buffer.
        """
        # Get the absolute TD errors and add epsilon
        td_errors = np.abs(self.get_experience_errors()) + epsilon

        # Raise the absolute TD errors to the power of prioritized_replay_alpha
        priorities = td_errors**self.prioritized_replay_alpha

        # Normalize priorities
        priorities = priorities / np.sum(priorities)

        return priorities

    def _get_experience_samples_from_indexes(
        self, indexes: np.ndarray[int]
    ) -> np.ndarray[
        tuple[np.ndarray[float], int, float, np.ndarray[float], bool, float]
    ]:
        return [self.storage[idx] for idx in indexes]
