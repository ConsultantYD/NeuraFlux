import logging as log
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, unique

import numpy as np

from neuraflux.agency.control_utils import softmax
from neuraflux.global_variables import (
    LOG_ENTITY_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_SIM_T_KEY,
)
from neuraflux.time_ref import TimeInfo


@dataclass
class Policy(metaclass=ABCMeta):
    agent_uid: int
    name: str = "Policy"

    @abstractmethod
    def apply_logic(self, *args, **kwargs) -> dict[int, int]:
        pass

    def __repr__(self) -> str:
        return self.name

    def __call__(self, *args, **kwargs) -> dict[int, int]:
        return self.apply_logic(*args, **kwargs)


@dataclass
class PolicyRandom(Policy):
    agent_uid: int
    name: str = "Random Policy"

    def apply_logic(
        self,
        n_controllers: int,
        action_size: int,
        time_info: None | TimeInfo = None,
    ) -> dict[int, int]:
        control_dict: dict[int, int] = {}

        for c in range(n_controllers):
            control_key = c + 1
            control_dict[control_key] = np.random.randint(action_size)

        # Log
        log.debug(
            {
                LOG_SIM_T_KEY: None if time_info is None else time_info.t,
                LOG_ENTITY_KEY: f"Agent({self.agent_uid})",
                LOG_METHOD_KEY: "apply_policy_random",
                LOG_MESSAGE_KEY: f"Random policy returning controls {str(control_dict)}",
            }
        )

        return control_dict


@dataclass
class PolicyQMax(Policy):
    agent_uid: int
    name: str = "Q-Max Policy"
    epsilon: float = 0.0
    min_probability: float = 0.05
    boltzmann: bool = False

    def apply_logic(
        self,
        q_factors: np.ndarray,
        possible_actions: list[int],
        time_info: None | TimeInfo = None,
    ) -> dict[int, int]:
        # Make sure epsilon is between 0 and 1
        epsilon = np.clip(self.epsilon, 0, 1)

        control_dict: dict[int, int] = {}

        # Loop over controllers and their Q-values to generate control data
        for c, q_tensor in enumerate(q_factors):  # c = controller
            # Take mean of all objectives if there are multiple
            q_values = np.mean(q_tensor[0, :, :], axis=0)
            control_key = c + 1
            # Random action probability
            if np.random.rand() < epsilon:  # Random action
                control_dict[control_key] = int(np.random.choice(possible_actions))
                policy_output = "epsilon random"
            # Else, greedy action
            else:
                # Boltzmann sampling mode
                if self.boltzmann:
                    softmax_probabilities = softmax(q_values)
                    # Ensure each probability is at least min_probability
                    probabilities = np.maximum(
                        softmax_probabilities, self.min_probability
                    )

                    # Normalize the probabilities to sum to 1
                    probabilities /= np.sum(probabilities)
                    control_int = np.random.choice(possible_actions, p=probabilities)
                    control_dict[control_key] = int(control_int)
                    policy_output = "boltzmann sampling"
                # Classical argmax mode
                else:
                    control_dict[control_key] = int(np.argmax(q_values))
                    policy_output = "argmax q-values"
            # Log
            log.debug(
                {
                    LOG_SIM_T_KEY: None if time_info is None else time_info.t,
                    LOG_ENTITY_KEY: f"Agent({self.agent_uid})",
                    LOG_METHOD_KEY: "apply_policy_q_max",
                    LOG_MESSAGE_KEY: f"Q-Max policy for controller {c+1} (eps={epsilon:.2f}) using {policy_output} returning controls {str(control_dict)} based on {q_factors}",
                }
            )

        # Delete unused variables and force garbage collection
        del q_factors

        return control_dict


@unique
class PoliciesEnum(Enum):
    Q_MAX_POLICY = PolicyQMax
    RANDOM_POLICY = PolicyRandom

    @classmethod
    def list_policies(cls) -> list[str]:
        # List all policy names in the enum, sorted alphabetically
        policies_list = list(map(lambda policy: policy.value().name, cls))
        policies_list.sort()
        return policies_list

    @classmethod
    def get_policy_class(cls, policy_name: str) -> Policy:
        # Iterate over all policies in the enum
        for policy in cls:
            # Match based on the name (case insensitive)
            if (
                policy.value().name == policy_name
                or policy.value().name.lower() == policy_name.lower()
            ):
                return policy.value
        # Raise an error if no matching policy is found
        raise ValueError(f"No policy found with name: {policy_name}")
