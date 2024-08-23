from neuraflux.geography import CityEnum
from neuraflux.local_typing import UidType
from neuraflux.schemas.agency import AgentConfig
from neuraflux.schemas.asset_config import (
    BuildingConfig,
    ElectricVehicleConfig,
    EnergyStorageConfig,
)

from .base import BaseSchema

# TODO: Add future asset configurations here
AssetConfigTypes = EnergyStorageConfig | BuildingConfig | ElectricVehicleConfig


class SimulationGeographicalConfig(BaseSchema):
    city: CityEnum = CityEnum.TORONTO

    class Config:
        use_enum_values = True


class SimulationTimeConfig(BaseSchema):
    start_time: str = "2023-01-01_00-00-00"
    end_time: str = "2023-02-01_00-00-00"
    step_size_s: int = 300


class SimulationDataConfig(BaseSchema):
    base_dir: str = "Data Module"


class SimulationConfig(BaseSchema):
    agents: dict[UidType, AgentConfig]
    assets: dict[UidType, AssetConfigTypes]
    directory: str = "DefaultSimulation"
    geography: SimulationGeographicalConfig
    seed: int = 42
    time: SimulationTimeConfig
