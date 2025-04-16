from typing import Any

from neuraflux.global_variables import (
    CONTROL_KEY,
    OAT_KEY,
    POWER_KEY,
    TIMESTAMP_KEY,
)

from .base import BaseSchema


class AssetConfig(BaseSchema):
    asset_type: str
    core_variables: list[str] = [
        TIMESTAMP_KEY,
        CONTROL_KEY,
        POWER_KEY,
        OAT_KEY,
    ]
    initial_state_dict: dict[str, Any] = {}
    n_controls: int = 1


# ENERGY STORAGE
class EnergyStorageConfig(AssetConfig):
    asset_type: str = "energy storage"
    capacity_kwh: float = 500.0
    control_power_mapping: dict[int, float] = {0: -100, 1: 0, 2: 100}
    efficiency_in: float = 1.0
    efficiency_out: float = 1.0
    decay_factor: float = 1.0
    tracked_variables: list[str] = ["internal_energy"]


# BUILDING ASSET
class BuildingConfig(AssetConfig):
    asset_type: str = "commercial building"
    dt: int = 5
    n_controls: int = 3
    control_power_mapping: dict[int, float] = {
        0: 40,  # Cooling Stage 2
        1: 20,  # Cooling Stage 1
        2: 0,  # Control Off
        3: 20,  # Heating Stage 1
        4: 40,  # Heating Stage 2
    }
    tracked_variables: list[str] = [
        "temperature",
        "hvac",
        "cool_setpoint",
        "heat_setpoint",
        "occupancy",
    ]
    initial_state_dict: dict[str, Any] = {
        "temperature": [21.0, 21.0, 21.0],
        "hvac": [0, 0, 0],
    }
    occ_times: tuple[int, int] = (8, 18)
    occ_setpoints: tuple[float, float] = (20.0, 22.0)
    unocc_setpoints: tuple[float, float] = (16.0, 26.0)


class ElectricVehicleConfig(AssetConfig):
    asset_type: str = "electric vehicle"
    capacity_kwh: float = 75.0
    control_power_mapping: dict[int, float] = {0: -25, 1: 0, 2: 25}
    tracked_variables: list[str] = ["internal_energy", "availability"]
    initial_state_dict: dict[str, Any] = {"internal_energy": 75, "availability": 1}
