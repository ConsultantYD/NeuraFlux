import datetime as dt
from typing import Any, Dict, Union

from .base import BaseSchema


# ----------------------------------------------------------------------------
# DEFAULT STATE WITH CORE VARIABLES
# ----------------------------------------------------------------------------
class AssetInternalState(BaseSchema):
    timestamp: Union[int, dt.datetime]

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> "AssetInternalState":
        return cls(**input_dict)


# ----------------------------------------------------------------------------
# ENERGY STORAGE STATE VARIABLES
# ----------------------------------------------------------------------------
class EnergyStorageInternalState(AssetInternalState):
    internal_energy: float
