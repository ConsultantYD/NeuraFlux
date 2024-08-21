import datetime as dt
import numpy as np

from neuraflux.agency.ddqn import DDQNPREstimator
from neuraflux.agency.replay_buffer import ReplayBuffer


def final_tuning_loop(
    replay_buffer_real: ReplayBuffer,
    replay_buffer_sim: ReplayBuffer,
    q_estimator: DDQNPREstimator,
    learning_rate: float = 5e-3,
    n_fit_epochs: int = 5,
    reduce_lr_patience: int = 2,
    no_improvement_patience: int = 9,
    min_improvement: int = 0.1,
):
    # Initialize variables
    best_q_estimator = q_estimator.copy()
    real_td_errors_list = []
    sim_td_errors_list = []

    # Get simulation replay buffer length
    # replay_buffer_sim_len = len(replay_buffer_sim)
    replay_buffer_real_len = len(replay_buffer_real)

    # Get all real experiences to track global performance
    all_real_exp, _, _ = get_info_from_replay_buffer(replay_buffer_real)
    all_sim_exp, _, _ = get_info_from_replay_buffer(replay_buffer_sim)

    td_errors_rmse_real = q_estimator.compute_td_errors_rmse(
        all_real_exp, aggregate=False
    )
    td_errors_rmse_sim = q_estimator.compute_td_errors_rmse(
        all_sim_exp, aggregate=False
    )

    replay_buffer_real.update_td_errors(td_errors=td_errors_rmse_real)
    replay_buffer_sim.update_td_errors(td_errors=td_errors_rmse_sim)

    # Get experience samples (e_real and e_sim) from replay buffer
    (batch_exp_sim, priorities_sim, _) = get_info_from_replay_buffer(
        replay_buffer=replay_buffer_sim,
        sampling_size=None,
    )

    (batch_exp_real, priorities_real, _) = get_info_from_replay_buffer(
        replay_buffer=replay_buffer_real,
        sampling_size=None,
    )

    # Combine both experiences by concatenating them
    batch_exp = []
    for i in range(len(batch_exp_sim)):
        batch_exp.append(np.concatenate((batch_exp_sim[i], batch_exp_real[i])))
    batch_exp = tuple(batch_exp)
    # priorities = np.concatenate((priorities_sim, priorities_real))

    # Compute initial real error for both this sample and global
    # err_real_0 = np.mean(
    #    q_estimator.compute_td_errors_rmse(batch_exp_real, aggregate=False)
    # )
    global_err_real = np.mean(
        q_estimator.compute_td_errors_rmse(all_real_exp, aggregate=False)
    )
    global_err_sim = np.mean(
        q_estimator.compute_td_errors_rmse(all_sim_exp, aggregate=False)
    )
    global_err = (global_err_real + global_err_sim) / 2
    real_td_errors_list.append(global_err_real)
    sim_td_errors_list.append(global_err_sim)

    # Fitting loop
    print("FINE-TUNING LOOP: ")
    no_improvement_counter = 0
    while True:
        # Fit Q estimator
        q_estimator.train(
            experience=batch_exp_real,
            replay_buffer_len=replay_buffer_real_len,
            learning_rate=learning_rate,
            priorities=priorities_real,
            n_fit_epochs=n_fit_epochs,
        )

        # Track global performance and update priorities
        all_real_exp, _, _ = get_info_from_replay_buffer(replay_buffer_real)
        all_sim_exp, _, _ = get_info_from_replay_buffer(replay_buffer_sim)

        td_errors_rmse_real = q_estimator.compute_td_errors_rmse(
            all_real_exp, aggregate=False
        )
        td_errors_rmse_sim = q_estimator.compute_td_errors_rmse(
            all_sim_exp, aggregate=False
        )

        replay_buffer_real.update_td_errors(td_errors=td_errors_rmse_real)
        replay_buffer_sim.update_td_errors(td_errors=td_errors_rmse_sim)

        # Update best Q estimator if global real error improved
        previous_global_err = global_err
        global_err_real = np.mean(td_errors_rmse_real)
        global_err_sim = np.mean(td_errors_rmse_sim)
        global_err = (global_err_real + global_err_sim) / 2
        real_td_errors_list.append(global_err_real)
        sim_td_errors_list.append(global_err_sim)

        print(f"    Global error: {global_err}")

        if previous_global_err - global_err >= min_improvement:
            best_q_estimator = q_estimator.copy()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= no_improvement_patience:
                break
            if no_improvement_counter % reduce_lr_patience == 0:
                q_estimator = best_q_estimator
                learning_rate /= 2

    return best_q_estimator, real_td_errors_list, sim_td_errors_list


def inner_simulation_training_loop(
    replay_buffer_real: ReplayBuffer,
    replay_buffer_sim: ReplayBuffer,
    q_estimator: DDQNPREstimator,
    sampling_size: int | None = None,
    inner_perc_threshold: float = 0.5,
    max_inner_iter: int = 10,
    learning_rate: float = 1e-3,
    n_fit_epochs: int = 5,
    no_improvement_patience: int = 2,
):
    # Initialize variables
    best_q_estimator = q_estimator.copy()
    real_td_errors_list = []
    sim_td_errors_list = []

    # Get simulation replay buffer length
    replay_buffer_sim_len = len(replay_buffer_sim)
    replay_buffer_real_len = len(replay_buffer_real)

    # Get all real experiences to track global performance
    all_real_exp, _, _ = get_info_from_replay_buffer(replay_buffer_real)
    all_sim_exp, _, _ = get_info_from_replay_buffer(replay_buffer_sim)

    td_errors_rmse_real = q_estimator.compute_td_errors_rmse(
        all_real_exp, aggregate=False
    )
    td_errors_rmse_sim = q_estimator.compute_td_errors_rmse(
        all_sim_exp, aggregate=False
    )

    replay_buffer_real.update_td_errors(td_errors=td_errors_rmse_real)
    replay_buffer_sim.update_td_errors(td_errors=td_errors_rmse_sim)

    # Get experience samples (e_real and e_sim) from replay buffer
    print(f"  Sampling {sampling_size} experiences from both PER.")
    (batch_exp_sim, priorities_sim, _) = get_info_from_replay_buffer(
        replay_buffer=replay_buffer_sim,
        sampling_size=sampling_size,
    )

    (batch_exp_real, priorities_real, _) = get_info_from_replay_buffer(
        replay_buffer=replay_buffer_real,
        sampling_size=sampling_size,
    )

    # Combine both experiences by concatenating them
    batch_exp = []
    for i in range(len(batch_exp_sim)):
        batch_exp.append(np.concatenate((batch_exp_sim[i], batch_exp_real[i])))
    batch_exp = tuple(batch_exp)
    priorities = np.concatenate((priorities_sim, priorities_real))

    # Compute initial real error for both this sample and global
    err_real_0 = np.mean(
        q_estimator.compute_td_errors_rmse(batch_exp_real, aggregate=False)
    )
    global_err_real = np.mean(
        q_estimator.compute_td_errors_rmse(all_real_exp, aggregate=False)
    )
    global_err_sim = np.mean(
        q_estimator.compute_td_errors_rmse(all_sim_exp, aggregate=False)
    )
    real_td_errors_list.append(global_err_real)
    sim_td_errors_list.append(global_err_sim)

    # Fitting loop
    print("  GENERALIZATION LOOP: ")
    no_improvement_counter = 0
    for j in range(max_inner_iter):
        # Fit Q estimator
        q_estimator.train(
            experience=batch_exp,
            replay_buffer_len=replay_buffer_sim_len + replay_buffer_real_len,
            learning_rate=learning_rate,
            priorities=priorities,
            n_fit_epochs=n_fit_epochs,
        )

        # Track global performance and update priorities
        all_real_exp, _, _ = get_info_from_replay_buffer(replay_buffer_real)
        all_sim_exp, _, _ = get_info_from_replay_buffer(replay_buffer_sim)

        td_errors_rmse_real = q_estimator.compute_td_errors_rmse(
            all_real_exp, aggregate=False
        )
        td_errors_rmse_sim = q_estimator.compute_td_errors_rmse(
            all_sim_exp, aggregate=False
        )

        replay_buffer_real.update_td_errors(td_errors=td_errors_rmse_real)
        replay_buffer_sim.update_td_errors(td_errors=td_errors_rmse_sim)

        # Update best Q estimator if global real error improved
        previous_global_err_real = global_err_real
        global_err_real = np.mean(td_errors_rmse_real)
        global_err_sim = np.mean(td_errors_rmse_sim)
        real_td_errors_list.append(global_err_real)
        sim_td_errors_list.append(global_err_sim)

        print(f"    > Global errors: {global_err_real} | {global_err_sim}")

        if global_err_real < previous_global_err_real:
            # print("        Improving best Q estimator !")
            best_q_estimator = q_estimator.copy()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Break if no improvement anymore
        if no_improvement_counter == no_improvement_patience:
            break

        # Break condition if improved beyond specific percentage
        err_real_j = q_estimator.compute_td_errors_rmse(batch_exp_real, aggregate=False)
        # TODO: Standardize this with the aggregate = True approach as well
        err_real_j = np.mean(err_real_j)
        print(f"    > Real sample error: {err_real_j}")
        if err_real_j / err_real_0 < inner_perc_threshold:
            print("        >> Breaking inner loop, performance reached.")
            break

    return best_q_estimator, real_td_errors_list, sim_td_errors_list


def simple_training_loop(
    replay_buffer_real: ReplayBuffer,
    q_estimator: DDQNPREstimator,
    sampling_size: int
    | None = None,  # Number of experiences to sample from replay buffer
    n_sampling_iters: int = 5,  # Number of sampling and fitting iterations
    learning_rate: float = 2.5e-4,
    n_fit_epochs: int = 3,  # Tensorflow training epochs for each exp batch
    tf_batch_size: int = 32,
) -> tuple[DDQNPREstimator, ReplayBuffer, list[float], list[float]]:
    # Initialize variables
    best_q_estimator = q_estimator.copy()

    # Get simulation replay buffer length
    replay_buffer_real_len = len(replay_buffer_real)

    # Keep list of all experience to track global performance
    all_real_exp, _, _ = get_info_from_replay_buffer(replay_buffer_real)

    # Track global performance and update priorities
    t0 = dt.datetime.now(dt.UTC)
    initial_td_errors_rmse_global = q_estimator.compute_td_errors_rmse(
        all_real_exp, aggregate=True
    )
    tf = dt.datetime.now(dt.UTC)

    # Keep detail of rmse global evolution
    all_errors = [initial_td_errors_rmse_global]
    times_dict = {
        "training": [None],
        "batch_sampling": [None],
        "global_td_errors_calculation": [(tf - t0).total_seconds()],
        "td_error_per_update": [None],
    }

    # Fitting loop
    for _ in range(n_sampling_iters):
        # Get experience samples (e_real and e_sim) from replay buffer
        # print(f"  Sampling {sampling_size} experiences from both PER.")
        t0 = dt.datetime.now(dt.UTC)
        (batch_exp_real, priorities_real, _) = get_info_from_replay_buffer(
            replay_buffer=replay_buffer_real,
            sampling_size=sampling_size,
        )
        tf = dt.datetime.now(dt.UTC)
        times_dict["batch_sampling"].append((tf - t0).total_seconds())

        # Combine both experiences by concatenating them
        batch_exp = batch_exp_real
        priorities = priorities_real

        # Fit Q estimator
        t0 = dt.datetime.now(dt.UTC)
        q_estimator.train(
            experience=batch_exp,
            replay_buffer_len=replay_buffer_real_len,
            learning_rate=learning_rate,
            priorities=priorities,
            n_fit_epochs=n_fit_epochs,
            batch_size=tf_batch_size,
        )
        # Store training time
        tf = dt.datetime.now(dt.UTC)
        times_dict["training"].append((tf - t0).total_seconds())

        # Track global performance and update priorities
        t0 = dt.datetime.now(dt.UTC)
        td_errors_rmse_real = q_estimator.compute_td_errors_rmse(
            all_real_exp, aggregate=False
        )
        tf = dt.datetime.now(dt.UTC)
        times_dict["global_td_errors_calculation"].append((tf - t0).total_seconds())

        # Updated TD errors based on the new Q estimator
        t0 = dt.datetime.now(dt.UTC)
        replay_buffer_real.update_td_errors(td_errors=td_errors_rmse_real)
        tf = dt.datetime.now(dt.UTC)
        times_dict["td_error_per_update"].append((tf - t0).total_seconds())

        # Always update best Q estimator in simple training
        best_q_estimator = q_estimator.copy()

        # Track global performance and update priorities
        global_td_errors_rmse = q_estimator.compute_td_errors_rmse(
            all_real_exp, aggregate=True
        )

        all_errors.append(global_td_errors_rmse)

    # Delete unused variables and force garbage collection
    del (
        all_real_exp,
        td_errors_rmse_real,
        q_estimator,
        batch_exp,
        priorities,
    )

    return (best_q_estimator, replay_buffer_real, all_errors, times_dict)


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

    # Delete unused variables and force garbage collection
    del exp, output, replay_buffer

    return batch_exp, priorities, indexes


def compute_td_errors_rmse(td_errors):
    td_errors_rmse = np.sqrt(np.mean(np.square(td_errors)))

    # Delete unused variables and force garbage collection
    del td_errors

    return td_errors_rmse


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
