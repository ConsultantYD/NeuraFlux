from dataclasses import dataclass
from .base import BaseSchema
from enum import Enum, unique


# ----------------------------------------------------------------------------
# CONTROL TYPES
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DiscreteControl(BaseSchema):
    value: int

    # Validate value
    def __post_init__(self):
        # Value > 0
        if self.value < 0:
            raise ValueError(
                "DiscreteControl value must be greater than or equal to 0."
            )

        # Value is int
        if not isinstance(self.value, int):
            raise TypeError("DiscreteControl value must be an integer.")


@dataclass(frozen=True)
class ContinuousControl(BaseSchema):
    value: float


@unique
class PolicyEnum(Enum):
    Q_MAX = "q_max"
    RANDOM = "random"
