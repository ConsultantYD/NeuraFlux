import numpy as np

from neuraflux.agency.dqn import DDQNPREstimator
from neuraflux.agency.replay_buffer import ReplayBuffer


def simple_training_loop(
    replay_buffer: ReplayBuffer,
    q_estimator: DDQNPREstimator,
    n_sampling_iters: int = 10,  # Number of sampling and fitting iterations
    sampling_size: int | None = 256,  # Number of exp sampled from buffer
    learning_rate: float = 1e-3,  # Learning rate for optimizer
    tf_n_fit_epochs: int = 1,  # Training epochs for tf .fit for each exp
    tf_batch_size: int = 16,  # Batch size for tf .fit
) -> tuple[DDQNPREstimator, ReplayBuffer, list[float], list[float]]:
    # Get total number of experiences in the replay buffer
    rb_len = len(replay_buffer)

    # Keep list of all experience to track global performance
    all_exp, _, _ = get_info_from_replay_buffer(replay_buffer)

    # Track global performance and update priorities
    initial_td_errors_rmse_global = q_estimator.compute_td_errors_rmse(
        all_exp, aggregate=True
    )

    # Keep detail of rmse global evolution
    all_errors = [initial_td_errors_rmse_global]

    # Fitting loop
    for _ in range(n_sampling_iters):
        (batch_exp, priorities, _) = get_info_from_replay_buffer(
            replay_buffer=replay_buffer,
            sampling_size=sampling_size,
        )

        # Fit Q estimator
        q_estimator.train(
            experience=batch_exp,
            replay_buffer_len=rb_len,
            learning_rate=learning_rate,
            priorities=priorities,
            n_fit_epochs=tf_n_fit_epochs,
            batch_size=tf_batch_size,
        )

        td_errors_rmse_real = q_estimator.compute_td_errors_rmse(
            all_exp, aggregate=False
        )

        # Updated TD errors based on the new Q estimator
        replay_buffer.update_td_errors(td_errors=td_errors_rmse_real)

        # Track global performance and update priorities
        global_td_errors_rmse = q_estimator.compute_td_errors_rmse(
            all_exp, aggregate=True
        )

        all_errors.append(global_td_errors_rmse)

    # Delete unused variables and force garbage collection
    return q_estimator, replay_buffer, all_errors


def get_info_from_replay_buffer(
    replay_buffer: ReplayBuffer,
    sampling_size: int | None = None,
):
    # Get experience samples from replay buffer
    output = replay_buffer.get_prioritized_experience_samples(
        sampling_size=sampling_size
    )
    (exp, priorities, indexes) = output
    batch_exp = batch_experience(experiences=exp)

    return batch_exp, priorities, indexes


def batch_experience(experiences: tuple) -> tuple:
    # Create individual arrays for each experience category
    states, actions, rewards, next_states, dones, errors = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for exp in experiences:
        state, action, reward, next_state, done, err = exp
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        errors.append(err)

    batched_experience = (
        np.array(states),
        np.array(actions),
        np.array(rewards),
        np.array(next_states),
        np.array(dones),
        np.array(errors),
    )

    # Delete unused variables and force garbage collection
    del states, actions, rewards, next_states, dones, errors

    return batched_experience
