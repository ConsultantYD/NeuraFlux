from enum import Enum, unique

from neuraflux.assets.energy_storage import EnergyStorage
from neuraflux.assets.building import Building


@unique
class AvailableAssetsEnum(Enum):
    ENERGY_STORAGE = EnergyStorage
    COMMERCIAL_BUILDING = Building
    # ELECTRIC_VEHICLE: str = "electric vehicle"
    # EV_CHARGER: str = "electric vehicle charger"
    # RESIDENTIAL_BUILDING: str = "residential building"
    # SOLAR_PANEL: str = "solar panel"
    # WIND_TURBINE: str = "wind turbine"

    @classmethod
    def list_assets(cls) -> list[str]:
        asset_names = [i.value.NAME for i in cls]
        asset_names.sort()
        return asset_names

    @classmethod
    def get_asset_class_from_asset_name(cls, asset_name: str):
        for _, asset_enum in cls.__members__.items():
            # Get the class from the enum value
            asset_class = asset_enum.value
            # Check if it has NAME and if it matches
            if hasattr(asset_class, "NAME") and asset_class.NAME == asset_name:
                return asset_class
        raise ValueError(f"No asset found with name: {asset_name}")

    @classmethod
    def get_asset_config_class_from_asset_name(cls, asset_name: str):
        return cls.get_asset_class_from_asset_name(asset_name).CONFIG_CLASS

    @classmethod
    def from_string(cls, asset_name: str, **kwargs):
        return cls.get_asset_class_from_asset_name(asset_name)(**kwargs)

    @classmethod
    def get_asset_config_types(cls):
        return [Asset.value.CONFIG_CLASS for Asset in cls]
