import datetime as dt
import pandas as pd
from neuraflux.schemas.asset_config import EnergyStorageConfig
from neuraflux.assets.factory import AvailableAssetsEnum
from neuraflux.schemas.control import DiscreteControl

CONTROL_POWER_MAPPING = {0: -100, 1: 0, 2: 100}
INITIAL_STATE_DICT = {
    "internal_energy": 0,
}

ASSET_CONFIG = EnergyStorageConfig(
    control_power_mapping=CONTROL_POWER_MAPPING,
    capacity_kwh=100,
    initial_state_dict=INITIAL_STATE_DICT,
)

t = dt.datetime(2023, 1, 1, 1, 0, 0)

asset = AvailableAssetsEnum.from_string("energy storage", name="test_asset", timestamp=t, config=ASSET_CONFIG, outside_air_temperature=10)

for _ in range(20):
    #control = [DiscreteControl(0)]
    #asset.step(control=control, timestamp=t, outside_air_temperature=20)
    asset.auto_step(timestamp=t, outside_air_temperature=20)
    t += dt.timedelta(minutes=5)
    
print(asset.get_historical_data(nan_padding=False))