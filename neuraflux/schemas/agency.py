from enum import Enum, unique

from .base import BaseSchema


# ----------------------------------------------------------------------------
# DATA SCALING SCHEMA
# ----------------------------------------------------------------------------
class ScalingMetadata(BaseSchema):
    min_value: float | int | None = None
    max_value: float | int | None = None
    min_sampled_value: float | int | None = None
    max_sampled_value: float | int | None = None
    scalable: bool = False


# ----------------------------------------------------------------------------
# REINFORCEMENT LEARNING CONFIGS
# ----------------------------------------------------------------------------
class RLConfig(BaseSchema):
    # General states and actions
    state_signals: list[str] | None = None
    action_size: int | None = None  # Number of possible actions
    n_controllers: int | None = None  # Number of controllers (>1 is multi-agent)
    history_length: int = 3  # Number of time steps to consider
    # Features
    add_hourly_time_features_to_state: bool = True
    add_daily_time_features_to_state: bool = True
    add_weekly_time_features_to_state: bool = False
    add_monthly_time_features_to_state: bool = False
    # Learning
    discount_factor: float = 0.99  # Discount factor for future rewards
    n_target_updates: int = 20  # Number of major target network update loops
    n_sampling_iters: int = 20  # Number of times to fit the model before target update
    experience_sampling_size: int = 500  # n of exp from PER at each sampling iter
    # Tensorflow training
    learning_rate: float = 0.00025  # Gradient descent learning rate in fit
    n_fit_epochs: int = 5  # Number of tensorflow epochs to fit the DQN targets
    tf_batch_size: int = 32  # Tensorflow training batch size
    # Experience Replay
    replay_buffer_size: int = 8500
    prioritized_replay_alpha: float = 0.6
    # prioritized_replay_beta0: float = 0.4
    # prioritized_replay_eps: float = 1e-6


# ----------------------------------------------------------------------------
# AGENT CONFIGS
# ----------------------------------------------------------------------------
@unique
class SignalTags(str, Enum):
    STATE: str = "X"
    CONTROL: str = "U"
    EXOGENOUS: str = "W"
    OBSERVATION: str = "O"
    RL_STATE: str = "S"  # Signal to use in the RL state


class SignalInfo(BaseSchema):
    tags: list[SignalTags] = []
    temporal_knowledge: int = 0
    min_value: float | int | None = None
    max_value: float | int | None = None
    scalable: bool = False


class AgentControlConfig(BaseSchema):
    n_controllers: int = 1
    n_trajectory_samples: int = 1
    trajectory_length: int = 6
    reinforcement_learning: RLConfig


class AgentDataConfig(BaseSchema):
    control_power_mapping: dict[int, float]
    tracked_signals: list[str]
    signals_info: dict[str, SignalInfo]


class AgentConfig(BaseSchema):
    control: AgentControlConfig
    data: AgentDataConfig
    product: str
    tariff: str
