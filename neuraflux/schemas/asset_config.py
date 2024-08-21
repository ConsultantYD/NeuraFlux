from enum import Enum, unique
from typing import Any

from pydantic import field_validator

from neuraflux.global_variables import (
    CONTROL_KEY,
    OAT_KEY,
    POWER_KEY,
    TIMESTAMP_KEY,
)

from .base import BaseSchema


@unique
class AvailableAssetsEnum(str, Enum):
    GENERIC: str = "generic"
    COMMERCIAL_BUILDING: str = "commercial building"
    ELECTRIC_VEHICLE: str = "electric vehicle"
    EV_CHARGER: str = "electric vehicle charger"
    ENERGY_STORAGE: str = "energy storage"
    RESIDENTIAL_BUILDING: str = "residential building"
    SOLAR_PANEL: str = "solar panel"
    WIND_TURBINE: str = "wind turbine"

    @classmethod
    def get_client_facing_name_mapping(cls) -> dict[str, str]:
        mapping_dict = {
            cls.GENERIC: "Generic Asset",
            cls.COMMERCIAL_BUILDING: "🏢 Commercial Building",
            cls.ELECTRIC_VEHICLE: "🚙 Electric Vehicle",
            cls.EV_CHARGER: "🔌 Electric Vehicle Charger",
            cls.ENERGY_STORAGE: "🔋 Energy Storage",
            cls.RESIDENTIAL_BUILDING: "🏠 Residential Building",
            cls.SOLAR_PANEL: "☀️ Solar Panel",
            cls.WIND_TURBINE: "💨 Wind Turbine",
        }
        return mapping_dict

    @classmethod
    def list_assets(cls) -> list[str]:
        asset_list = [asset.value for asset in cls]
        asset_list.remove(cls.GENERIC)
        asset_list.sort()
        return asset_list

    @classmethod
    def list_assets_for_frontend(cls) -> list[str]:
        frontend_asset_list = [
            cls.get_client_facing_name_mapping()[asset.value]
            for asset in cls
            if asset.value != cls.GENERIC
        ]
        return frontend_asset_list

    @classmethod
    def get_asset_from_client_facing_name(cls, client_facing_name: str) -> str:
        for asset, name in cls.get_client_facing_name_mapping().items():
            if name == client_facing_name:
                return asset
        raise ValueError("Client facing name not found.")


class AssetComponent(BaseSchema):
    name: str

    def __str__(self):
        return self.name


class Dishwasher(AssetComponent):
    name: str = "Dishwasher"


class HVACBaseboard(AssetComponent):
    name: str = "HVAC - Baseboard"


class HVACRoofTopUnit(AssetComponent):
    name: str = "HVAC - Roof Top Unit"


class Lighting(AssetComponent):
    name: str = "Lighting"


class Oven(AssetComponent):
    name: str = "Oven"


class WaterHeater(AssetComponent):
    name: str = "Water Heater"


class AssetComponentEnum(str, Enum):
    DISHWASHER: str = Dishwasher()
    HVAC_BASEBOARD: str = HVACBaseboard()
    HVAC_ROOF_TOP_UNIT: str = HVACRoofTopUnit()
    LIGHTING: str = Lighting()
    OVEN: str = Oven()
    WATER_HEATER: str = WaterHeater()

    @classmethod
    def list_components(cls) -> list[str]:
        component_list = [component.name for component in cls]
        component_list.sort()
        return component_list

    @classmethod
    def get_component_from_name(cls, component_name: str) -> str:
        for component in cls:
            if component.value == component_name:
                return component
        raise ValueError("Component name not found.")


class AssetConfig(BaseSchema):
    asset_type: AvailableAssetsEnum = AvailableAssetsEnum.GENERIC
    core_variables: list[str] = [
        TIMESTAMP_KEY,
        CONTROL_KEY,
        POWER_KEY,
        OAT_KEY,
    ]
    initial_state_dict: dict[str, Any] = {}
    n_controls: int = 1


# BUILDING ASSET
class BuildingConfig(AssetConfig):
    asset_type: AvailableAssetsEnum = AvailableAssetsEnum.COMMERCIAL_BUILDING
    control_power_mapping: dict[int, float] = {
        0: 20,  # Cooling Stage 2
        1: 10,  # Cooling Stage 1
        2: 0,  # Control Off
        3: 10,  # Heating Stage 1
        4: 20,  # Heating Stage 2
    }
    tracked_variables: list[str] = [
        "temperature",
        "hvac",
        "cool_setpoint",
        "heat_setpoint",
        "occupancy",
    ]
    n_controls: int = 3
    initial_state_dict: dict[str, Any] = {
        "temperature": [21.0, 21.0, 21.0],
        "hvac": [0, 0, 0],
    }
    dt: int = 5
    occ_times: tuple[int, int] = (8, 18)
    occ_setpoints: tuple[float, float] = (20.0, 22.0)
    unocc_setpoints: tuple[float, float] = (16.0, 26.0)


# ENERGY STORAGE ASSET
class EnergyStorageEfficiencyLimits:
    UPPER_LIMIT: float = 1.0
    LOWER_LIMIT: float = 0.0


class EnergyStorageEfficiency(BaseSchema):
    value: float = 1.0

    @field_validator("value")
    def check_value(cls, v: float) -> float:  # pylint: disable=no-self-argument
        if (
            v > EnergyStorageEfficiencyLimits.UPPER_LIMIT
            or v < EnergyStorageEfficiencyLimits.LOWER_LIMIT
        ):
            raise ValueError("Efficiency must be between 0 and 1.")
        return v


class EnergyStorageConfig(AssetConfig):
    asset_type: AvailableAssetsEnum = AvailableAssetsEnum.ENERGY_STORAGE
    capacity_kwh: float = 500.0
    control_power_mapping: dict[int, float] = {1: -100, 2: 0, 3: 100}
    efficiency_in: EnergyStorageEfficiency = EnergyStorageEfficiency()
    efficiency_out: EnergyStorageEfficiency = EnergyStorageEfficiency()
    decay_factor: float = 1.0
    tracked_variables: list[str] = ["internal_energy"]


class ElectricVehicleConfig(AssetConfig):
    asset_type: AvailableAssetsEnum = AvailableAssetsEnum.ELECTRIC_VEHICLE
    capacity_kwh: float = 75.0
    control_power_mapping: dict[int, float] = {0: -25, 1: 0, 2: 25}
    tracked_variables: list[str] = ["internal_energy", "availability"]
    initial_state_dict: dict[str, Any] = {"internal_energy": 75, "availability": 1}
