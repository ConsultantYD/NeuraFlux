from neuraflux.assets.factory import AvailableAssetsEnum
from neuraflux.geography import CityEnum
from neuraflux.local_typing import UidType
from neuraflux.schemas.agency import AgentConfig

from .base import BaseSchema


class SimulationGeographicalConfig(BaseSchema):
    city: CityEnum = CityEnum.TORONTO

    class Config:
        use_enum_values = True


class SimulationTimeConfig(BaseSchema):
    start_time: str = "2023-01-01T00:00:00"
    end_time: str = "2023-02-01T00:00:00"
    step_size_s: int = 300


class SimulationDataConfig(BaseSchema):
    base_dir: str = "Data Module"


class SimulationConfig(BaseSchema):
    agents: dict[UidType, AgentConfig]
    assets: dict[UidType, object]
    agent_save_freq_cron: str = "0 0 * * *"
    directory: str = "DefaultSimulation"
    geography: SimulationGeographicalConfig
    seed: int = 42
    time: SimulationTimeConfig

    # NOTE: Method to choose the correct asset configuration based on the asset type
    @classmethod
    def from_custom_dict(cls, data: dict):
        self = cls.model_validate(data)
        for asset_name, asset_config in data["assets"].items():
            AssetConfig = AvailableAssetsEnum.get_asset_config_class_from_asset_name(
                asset_config["asset_type"]
            )
            asset_config = AssetConfig.model_validate(asset_config)
            self.assets[asset_name] = asset_config
        return self
