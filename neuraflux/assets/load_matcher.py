import datetime as dt

import numpy as np

from neuraflux.schemas.asset_config import LoadMatcherConfig
from neuraflux.assets.base_asset import Asset
from neuraflux.schemas.control import DiscreteControl


class LoadMatcherValidator(Asset):
    CONFIG_CLASS = LoadMatcherConfig
    NAME = "load matcher (validator)"

    def step(
        self,
        control: list[DiscreteControl],
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> float:
        # Controller choses directly the provided power
        self.power = control[0].value

        # Move load stochasticly by 1
        self.load += np.random.choice([-1, 0, 1])
        self.load = np.clip(self.load, 0, 4)

        # Run base class to store variables of interest
        super().step(control, timestamp, outside_air_temperature)
        return self.power

    def get_auto_control(self, *args, **kwargs) -> list[DiscreteControl]:
        return [DiscreteControl(value=np.random.randint(0, 5))]
