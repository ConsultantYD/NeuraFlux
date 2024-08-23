import datetime as dt
import pandas as pd
from neuraflux.assets.building import Building
from neuraflux.assets.electric_vehicle import ElectricVehicle
from neuraflux.assets.energy_storage import EnergyStorage

# General
IndexType = int | dt.datetime
SignalType = dict[str, (float | int | str)]
UidType = str

# Assets
AssetType = Building | ElectricVehicle | EnergyStorage

# Data Module
DBSignalType = dict[UidType, dict[IndexType, SignalType]]
DBScalingDictsType = dict[UidType, dict[str, tuple[float, float]]]
DBTrajectoriesType = dict[UidType, dict[IndexType, list[pd.DataFrame]]]
