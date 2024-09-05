import logging as log
import os
from copy import copy
from dataclasses import dataclass
from shutil import rmtree
import traceback

import dill
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from neuraflux.global_variables import (
    LOG_ENTITY_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_SIM_T_KEY,
)


@dataclass
class DDQNPREstimator:
    state_size: int
    action_size: int
    sequence_len: int
    n_rewards: int = 1
    n_controllers: int = 1
    learning_rate: float = 2.5e-4
    discount_factor: float = 0.99

    def __post_init__(self) -> None:
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def __call__(self, *args, **kwargs):
        return self.forward_pass(*args, **kwargs)

    def forward_pass(
        self,
        states: np.ndarray,
        target_model: bool = False,
        lite_model: bool = False,
    ) -> np.ndarray:
        if lite_model:
            try:
                return self.lite_predict(states)
            except:
                print("Inference with lite model failed. Switching to full model.")
                print(traceback.format_exc())
        tf.compat.v1.get_default_graph().finalize()
        # Copy state and convert to tensor
        states = states.copy()
        # states = tf.convert_to_tensor(states.copy())

        # Define model and apply forward pass
        model = self.target_model if target_model else self.model
        # outputs = model.predict(states, verbose=0) # Outputs numpy
        outputs = model(states)

        # Ensure outputs are a list, even for a single controller
        if not isinstance(outputs, list):
            outputs = [outputs]

        # Convert default tensor outputs to numpy arrays
        outputs = [o.numpy() for o in outputs]

        # Delete unused variables and force garbage collection
        del states, model

        return outputs

    def lite_predict(self, x):
        if x.shape[0] > 1:
            raise ValueError("Lite model only supports batch size of 1.")

        # Create lite model if it doesn't exist
        if not hasattr(self, "lite_model"):
            self.generate_tflite_model()

        # Run the model with TensorFlow Lite
        interpreter = tf.lite.Interpreter(model_content=self.lite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

        result = [
            interpreter.get_tensor(output_details[i]["index"])
            for i in range(self.n_controllers)
        ]

        # NOTE: TfLite fused Lstm kernel is stateful, so we need to reset
        # the states.
        # Clean up internal states.
        interpreter.reset_all_variables()
        return result

    def compute_td_errors(
        self,
        experience: tuple,
    ) -> list:
        (states, actions, rewards, next_states, dones, errors) = experience

        # Get the current Q-values (list of [batch_size, n_rewards, action_size] for each controller)
        q_values_current = self.forward_pass(states)

        # Get the Q-values for the next states from the target model
        q_values_next_target = self.forward_pass(next_states, target_model=True)

        # Initialize a list to hold the TD errors for each controller
        td_errors = [
            np.zeros(
                (states.shape[0], self.n_rewards, self.action_size)
            )  # n_samples x n_rewards x n_actions
            for _ in range(self.n_controllers)
        ]

        # Compute TD errors for each controller and reward
        for controller in range(self.n_controllers):
            for reward in range(self.n_rewards):
                # Compute the max Q-value for the next states for each reward
                q_max_next_target = np.max(
                    q_values_next_target[controller][:, reward, :], axis=1
                )

                # Compute the target Q-values
                q_target = rewards[
                    :, reward
                ] + self.discount_factor * q_max_next_target * (1 - dones)

                # Select the corresponding actions for the current controller
                actions_current_controller = actions[:, controller]

                # Compute the TD errors for all actions
                for action in range(self.action_size):
                    is_action_taken = actions_current_controller == action
                    for sample in range(states.shape[0]):
                        if is_action_taken[sample]:
                            td_errors[controller][sample, reward, action] = (
                                q_target[sample]
                                - q_values_current[controller][sample, reward, action]
                            )
                        else:
                            td_errors[controller][sample, reward, action] = (
                                0  # np.nan  # Default value for non-taken actions
                            )

        # Delete unused variables and force garbage collection
        del (
            states,
            actions,
            rewards,
            next_states,
            dones,
            errors,
            q_values_current,
            q_values_next_target,
        )

        return td_errors

    def compute_td_errors_rmse(self, experience, aggregate=True):
        td_errors = self.compute_td_errors(experience)
        # Check for invalid data and raise ValueError if found
        for controller_td_error in td_errors:
            if controller_td_error.size == 0:
                raise ValueError("Encountered an empty array in TD errors.")

        if aggregate:
            # Compute aggregated RMSE for each controller separately
            rmse_per_controller = [
                np.sqrt(np.nanmean(np.square(controller_td_error)))
                for controller_td_error in td_errors
            ]

            # Aggregate RMSE across all controllers
            aggregated_rmse = np.nanmean(rmse_per_controller)

            # Delete unused variables and force garbage collection
            del td_errors, rmse_per_controller

            return aggregated_rmse
        else:
            # Compute RMSE for each sample individually for each controller
            rmse_per_sample_per_controller = [
                np.sqrt(
                    np.nanmean(np.square(controller_td_error), axis=(1, 2))
                )  # Compute RMSE across rewards and actions
                for controller_td_error in td_errors
            ]

            # Sum the RMSEs element-wise across all controllers and divide by the number of controllers
            averaged_rmse_per_sample = np.nanmean(
                rmse_per_sample_per_controller, axis=0
            )

            # Delete unused variables and force garbage collection
            del td_errors, rmse_per_sample_per_controller

            # Return the averaged RMSEs for each sample
            return averaged_rmse_per_sample

    def train(
        self,
        experience: tuple,
        replay_buffer_len: int,
        priorities: np.ndarray,
        learning_rate: float = 2.5e-4,
        beta: float = 0.4,
        n_fit_epochs: int = 5,
        batch_size: int = 32,
    ):
        # Unpack the experience tuple
        (states, actions, rewards, next_states, dones, errors) = experience
        batch_size = states.shape[0]

        # Dimensions of each entry in the experience tuple
        # states is (batch_size, sequence_len, state_size)
        # actions was (batch_size,), is now (batch_size, n_controllers)
        # rewards was (batch_size,), is now (batch_size, n_rewards)
        # next_states is (batch_size, sequence_len, state_size)

        # Copy the states to avoid modifying the original
        states = states.copy()

        # Calculate the necessary targets
        # was (batch_size, n_actions), is now [(batch_size, n_rewards, n_actions), ...]
        # where the len of the list is n_controllers)
        targets = self.forward_pass(states)
        targets_next = self.forward_pass(next_states)
        targets_val = self.forward_pass(next_states, target_model=True)

        for target, target_next, target_val in zip(targets, targets_next, targets_val):
            for r in range(self.n_rewards):
                for i in range(batch_size):
                    max_a = np.argmax(target_next[i][r])

                    term_1 = rewards[i][r]
                    term_2 = (
                        self.discount_factor * target_val[i][r][max_a] * (1 - dones[i])
                    )
                    target[i][r][actions[i]] = term_1 + term_2

        # Log the first sample to confirm the data is correct
        log.debug(
            {
                LOG_SIM_T_KEY: None,
                LOG_ENTITY_KEY: "DDQNPREstimator",
                LOG_METHOD_KEY: "train",
                LOG_MESSAGE_KEY: f"First state: {states[0]} \n"
                f"First Q-factors: {self.forward_pass(states)[0][0]} \n"
                f"First reward: {rewards[0]} \n"
                f"First action: {actions[0]} \n"
                f"First max a (r=1): {np.argmax(targets_next[0][0])} \n"
                f"First target: {targets[0][0]} \n"
                f"First target_next: {targets_next[0][0]} \n"
                f"First target_val: {targets_val[0][0]} \n",
            }
        )

        # Compute importance sampling weights
        importance_sampling_weights = np.power(replay_buffer_len * priorities, -beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        # Compile model TODO: Add term if we need to reduce LR
        # if not self.model.compiled: # TF >=2.16
        if not self.model._is_compiled:  # TF <2.16
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    clipnorm=1.0,
                ),
                loss="huber",
            )
        # tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)

        # print(
        #    f"    Fitting DQN for {n_fit_epochs} epochs with learning rate {learning_rate}"
        # )
        tf.compat.v1.get_default_graph().finalize()
        self.model.fit(
            # tf.convert_to_tensor(states),
            # tf.convert_to_tensor(target),
            states,
            target,
            batch_size=batch_size,
            verbose=0,
            sample_weight=importance_sampling_weights,
            epochs=n_fit_epochs,
        )

        # Delete unused variables and force garbage collection
        del (
            states,
            actions,
            rewards,
            next_states,
            dones,
            errors,
            targets,
            targets_next,
            targets_val,
            importance_sampling_weights,
        )

    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def to_file(self, directory: str) -> None:
        # Create directory if it doesn't exist
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Save the model
        model_file = os.path.join(directory, "model.keras")
        self.model.save(model_file)

        # Save the lite model
        if hasattr(self, "lite_model"):
            lite_model_file = os.path.join(directory, "model.tflite")
            with open(lite_model_file, "wb") as f:
                f.write(self.lite_model)

        # Remove unserializable objects from the attributes
        model_temp = self.model
        target_model_temp = self.target_model
        del self.model
        del self.target_model

        # Save the attributes
        filepath = os.path.join(directory, "attributes.pkl")
        with open(filepath, "wb") as f:
            dill.dump(vars(self), f)

        # Restore attributes
        self.model = model_temp
        self.target_model = target_model_temp

        # Delete unused variables and force garbage collection
        del model_temp, target_model_temp

    @classmethod
    def from_file(cls, directory: str = "") -> "DDQNPREstimator":
        """Loads an estimator from the specified directory.

        directory (str): The directory to load the attributes from.
        """
        model_file = os.path.join(directory, "model.keras")
        filepath = os.path.join(directory, "attributes.pkl")
        with open(filepath, "rb") as f:
            internal_variables = dill.load(f)
        self = cls(
            internal_variables["state_size"],
            internal_variables["action_size"],
            internal_variables["sequence_len"],
        )
        self.__dict__.update(internal_variables)
        self.model = tf.keras.models.load_model(model_file)
        self.update_target_model()

        # If lite_model available
        lite_model_file = os.path.join(directory, "model.tflite")
        if os.path.exists(lite_model_file):
            with open(lite_model_file, "rb") as f:
                self.lite_model = f.read()

        return self

    def copy(self):
        self_copy = copy(self)
        self_copy.model = tf.keras.models.clone_model(self.model)
        self_copy.model.set_weights(self.model.get_weights())
        self_copy.target_model = tf.keras.models.clone_model(self.target_model)
        return self_copy

    def generate_tflite_model(self):
        run_model = tf.function(lambda x: self.model(x))

        # Fix the input size.
        BATCH_SIZE = 1
        STEPS = self.sequence_len
        INPUT_SIZE = self.state_size
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], self.model.inputs[0].dtype)
        )

        # model directory.
        MODEL_DIR = "temp_keras"
        self.model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

        converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        self.lite_model = converter.convert()

        # Clean up the model directory.
        rmtree(MODEL_DIR)

    # def _build_model(self) -> Model:
    #     # initializer = tf.keras.initializers.Zeros()
    #     input_layer = tf.keras.layers.Input(
    #         shape=(self.sequence_len, self.state_size)
    #     )

    #     x = tf.keras.layers.LSTM(
    #         128, activation="tanh", return_sequences=False
    #     )(input_layer)

    #     q_values_list = [
    #         tf.keras.layers.Dense(
    #             self.action_size,
    #             activation="linear",
    #         )(x)
    #         for _ in range(self.n_rewards)
    #     ]

    #     model = Model(inputs=[input_layer], outputs=q_values_list)
    #     model.compile(
    #         optimizer=tf.keras.optimizers.Adam(
    #             learning_rate=self.learning_rate, clipnorm=1
    #         ),
    #         loss="mse",
    #     )

    #     return model

    def _build_model(self) -> Model:
        # Clear the session to free up memory
        tf.keras.backend.clear_session()

        input_layer = tf.keras.layers.Input(shape=(self.sequence_len, self.state_size))

        # Shared layers
        x = tf.keras.layers.LSTM(
            128,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )(input_layer)

        # Output layer for all actions and rewards
        output_layers = []
        for _ in range(self.n_controllers):
            output_layer = tf.keras.layers.Dense(
                self.n_rewards * self.action_size,
                activation="linear",
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=0.01, mode="fan_avg", distribution="uniform"
                ),
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            )(x)
            # Reshape to [number of rewards, number of actions]
            output_reshaped = tf.keras.layers.Reshape(
                (self.n_rewards, self.action_size)
            )(output_layer)
            output_layers.append(output_reshaped)

        model = Model(inputs=[input_layer], outputs=output_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0,
            ),
            loss="huber",
        )
        return model
