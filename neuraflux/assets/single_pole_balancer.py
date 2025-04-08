import datetime as dt
import gymnasium as gym

import numpy as np

from neuraflux.schemas.asset_config import SinglePoleBalancerConfig
from neuraflux.assets.base_asset import Asset
from neuraflux.schemas.control import DiscreteControl


class SinglePoleBalancerValidator(Asset):
    CONFIG_CLASS = SinglePoleBalancerConfig
    NAME = "pole balancer (validator)"

    def __init__(self, *args, **kwargs):
        self.env = gym.make("CartPole-v1")
        s = self.env.reset()[0]
        self.up_counter = 0
        self.cart_position = s[0]
        self.cart_velocity = s[1]
        self.pole_angle = s[2]
        self.pole_velocity = s[3]

        super().__init__(*args, **kwargs)

    def step(
        self,
        control: list[DiscreteControl],
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> float:
        self.power = 0
        # Increment the environment
        action = control[0].value
        s, _, d, _, _ = self.env.step(action)

        if d:
            s = self.env.reset()[0]
            self.up_counter = 0
        else:
            self.up_counter += 1

        # Define internal variables
        self.cart_position = s[0]
        self.cart_velocity = s[1]
        self.pole_angle = s[2]
        self.pole_velocity = s[3]

        # Run base class to store variables of interest
        super().step(control, timestamp, outside_air_temperature)
        return self.power

    def get_auto_control(self, *args, **kwargs) -> list[DiscreteControl]:
        return [DiscreteControl(value=np.random.randint(0, 2))]
