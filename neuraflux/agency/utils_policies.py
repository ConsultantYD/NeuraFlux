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


def hvac_policy(
    temperatures: np.array,
    setpoints: tuple[float, float],
    q_values: list[np.array],
    epsilon: float,
    comfort_constraint: bool = True,
):
    """
    HVAC policy that selects actions based on temperature and Q-values.

    Parameters
    ----------
    temperatures : np.ndarray
        Current temperatures for each controller.
    setpoints : tuple[float, float]
        Setpoint values for the HVAC system.
    q_values : list[np.ndarray]
        Q-values for each controller.
    epsilon : float
        Probability of choosing a random action.
    comfort_constraint : bool
        Whether to apply comfort constraints.

    Returns
    -------
    np.ndarray
        Action probabilities for each state.
    """
    # Compute and apply the def Q-policy by default
    controls = np.array(q_policy(q_values, epsilon))

    if comfort_constraint:
        # Check for discomfort, and automatically set stage 1 if not present
        controls[(temperatures < setpoints[0]) & (controls < 3)] = 3
        controls[(temperatures > setpoints[1]) & (controls > 1)] = 1

    # TODO: Add power cap logic

    return controls.tolist()
