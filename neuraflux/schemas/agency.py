from enum import Enum, unique
from typing import Any, Literal

from pydantic import Field

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
    action_size: int  # Number of possible actions
    state_signals: list[str] | None = None
    n_controllers: int = 1  # Number of controllers (>1 is multi-agent)
    history_length: int = 3  # Number of time steps to consider
    discount_factor: float = 1.0  # Discount factor for future rewards
    # Features
    add_hourly_time_features_to_state: bool = True
    add_daily_time_features_to_state: bool = True
    add_weekly_time_features_to_state: bool = False
    add_monthly_time_features_to_state: bool = False


class RLTrainingConfig(BaseSchema):
    # Learning
    target_update_period: int = 20  # Number of major target network update loops
    n_fit_epochs: int = 10  # Number of times to fit the DQN to experience
    experience_sampling_size: int = 128  # n of exp from PER at each sampling iter
    # Tensorflow training
    learning_rate: float = 5e-4  # Gradient descent learning rate in fit
    gradient_clip: float = 0.5  # Gradient clipping value
    n_target_iterators: int = 20  # Number of times to fit the DQN to experience
    n_sampling_iters: int = 10  # Number of times to sample from the experience replay
    tf_batch_size: int = 32  # Batch size for Tensorflow training


class RealLearningConfig(BaseSchema):
    # General and orchestration
    enabled: bool = True  # Whether to enable real learning
    trigger_freq_cron: str = "0 0 * * 1"  # Training frequency
    # Training
    rl_training_config: RLTrainingConfig = RLTrainingConfig()


class SimLearningConfig(BaseSchema):
    # General and orchestration
    enabled: bool = True  # Whether to enable simulation training
    trigger_freq_cron: str = "0 0 * * *"  # Training frequency
    # Sampling and generating simulated trajectories
    # NOTE: n_traj = n_samples(~t) * n_traj_per_sample
    n_samples: int = 100  # Number of real timestamps to sample from
    n_traj_per_sample: int = 1  # Number of trajectories to generate at each sample
    trajectory_len: int = 18  # Length of each trajectory sampled and simulated
    policy: Literal["random_policy", "q_policy", "hvac_policy"] = "hvac_policy"
    policy_kwargs: dict[str, object] = {"epsilon": 0.5, "comfort_constraint": False}
    # Training
    rl_training_config: RLTrainingConfig = RLTrainingConfig()


# ----------------------------------------------------------------------------
# AGENT CONFIGS
# ----------------------------------------------------------------------------
@unique
class SignalTags(str, Enum):
    STATE: str = "X"  # State of the system
    CONTROL: str = "U"  # Control signal
    EXOGENOUS: str = "W"  # Exogenous signal
    OBSERVATION: str = "O"  # Useful observation
    RL_STATE: str = "S"  # Signal to use in the RL state


@unique
class SignalSource(str, Enum):
    ASSET: str = "asset"
    EXTERNAL_PROVIDER: str = "external_provider"
    PRODUCT: str = "product"


class SignalInfo(BaseSchema):
    min_value: float | int | None = None
    max_value: float | int | None = None
    scalable: bool = False
    source: SignalSource = SignalSource.ASSET
    tags: list[SignalTags] = []
    temporal_knowledge: tuple = (None, 0)


class ControlSelectionConfig(BaseSchema):
    enabled: bool = True
    policy: Literal["random_policy", "q_policy", "hvac_policy"] = "q_policy"
    policy_kwargs: dict[str, object] = {}


class AgentControlConfig(BaseSchema):
    n_controllers: int
    control_selection: dict[int, ControlSelectionConfig] = Field(
        default_factory=lambda: {
            0: ControlSelectionConfig(enabled=False),
            60 * 60 * 24 * 7: ControlSelectionConfig(
                policy="q_policy", policy_kwargs={"epsilon": 0.0}
            ),
        }
    )
    rl_config: RLConfig
    real_learning_configs: dict[int, RealLearningConfig] = Field(
        default_factory=lambda: {
            0: RealLearningConfig(enabled=False),
            60 * 60 * 24 * 7: RealLearningConfig(),
        }
    )
    real_replay_buffer_size: int = 10000
    sim_learning_configs: dict[int, SimLearningConfig] = Field(
        default_factory=lambda: {
            0: SimLearningConfig(enabled=False),
            60 * 60 * 24 * 1: SimLearningConfig(),
        }
    )
    sim_replay_buffer_size: int = 1000


class AgentDataConfig(BaseSchema):
    control_power_mapping: dict[int, float]
    tracked_signals: list[str]
    signals_info: dict[str, SignalInfo]
    memory_dump_freq_cron: str = "55 23 * * *"


class AgentConfig(BaseSchema):
    asset_metadata: dict[str, Any] = {
        "address": "123 Fake St, Anytown, CA",
        "timezone": "America/Toronto",
        "location": (43.7, -79.42, 0.0),
        "owner": "John Doe",
    }
    control: AgentControlConfig
    components_metadata: list[dict[str, Any]] = []
    data: AgentDataConfig
    product: str
    tariff: str
