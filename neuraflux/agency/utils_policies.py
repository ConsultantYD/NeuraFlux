import numpy as np


def random_policy(action_size: int, n_controllers: int):
    """
    Random policy that selects a random action.

    Returns
    -------
    int
        Randomly selected action.
    """
    return [int(np.random.randint(0, action_size)) for _ in range(n_controllers)]


def q_policy(q_values: list[np.array], epsilon: float):
    """
    Epsilon-greedy policy based on Q-values.

    Parameters
    ----------
    q_values : list[np.ndarray]
        Q-values for each controller.
    epsilon : float
        Probability of choosing a random action.

    Returns
    -------
    np.ndarray
        Action probabilities for each state.
    """
    n_controllers = len(q_values)
    action_size = q_values[0].shape[-1]
    if np.random.rand() <= epsilon:
        return [int(np.random.randint(0, action_size)) for _ in range(n_controllers)]
    return [int(np.argmax(q_values[c][-1].flatten())) for c in range(n_controllers)]
