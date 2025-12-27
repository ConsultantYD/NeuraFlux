from neuraflux.assets.factory import AvailableAssetsEnum
from neuraflux.geography import CityEnum
from neuraflux.local_typing import UidType
from neuraflux.schemas.agency import AgentConfig

from pydantic import ConfigDict, field_validator, model_validator

from neuraflux.schemas.asset_config import AssetConfig

from .base import BaseSchema


class SimulationGeographicalConfig(BaseSchema):
    model_config = ConfigDict(use_enum_values=True)
    city: CityEnum = CityEnum.TORONTO


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
    data: SimulationDataConfig = SimulationDataConfig()
    directory: str = "DefaultSimulation"
    geography: SimulationGeographicalConfig
    seed: int = 42
    time: SimulationTimeConfig

    @field_validator("assets", mode="before")
    @classmethod
    def _normalize_assets(cls, v: object) -> dict[UidType, AssetConfig]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError(
                "SimulationConfig.assets must be a dict of uid -> asset config"
            )

        normalized: dict[UidType, AssetConfig] = {}
        for uid, asset in v.items():
            if isinstance(asset, AssetConfig):
                normalized[uid] = asset
                continue
            if isinstance(asset, dict):
                asset_type = asset.get("asset_type")
                if not asset_type:
                    raise ValueError(
                        f"Asset '{uid}' config is missing required field 'asset_type'."
                    )
                AssetConfigClass = (
                    AvailableAssetsEnum.get_asset_config_class_from_asset_name(
                        asset_type
                    )
                )
                normalized[uid] = AssetConfigClass.model_validate(asset)
                continue
            raise TypeError(
                f"Asset '{uid}' must be an AssetConfig instance or a dict; got {type(asset)}."
            )

        return normalized

    @model_validator(mode="after")
    def _validate_agent_asset_uids_match(self) -> "SimulationConfig":
        agent_uids = set(self.agents.keys())
        asset_uids = set(self.assets.keys())
        if agent_uids == asset_uids:
            return self

        missing_assets = sorted(agent_uids - asset_uids)
        missing_agents = sorted(asset_uids - agent_uids)
        parts: list[str] = []
        if missing_assets:
            parts.append(f"missing assets for agents: {missing_assets}")
        if missing_agents:
            parts.append(f"missing agents for assets: {missing_agents}")
        raise ValueError(
            "SimulationConfig agent/asset UID mismatch: " + "; ".join(parts)
        )

    # NOTE: Method to choose the correct asset configuration based on the asset type
    @classmethod
    def from_custom_dict(cls, data: dict):
        return cls.model_validate(data)
