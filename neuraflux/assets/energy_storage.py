import datetime as dt

import numpy as np
from neuraflux.assets.base_asset import Asset
from neuraflux.schemas.control import DiscreteControl
from neuraflux.schemas.asset_config import EnergyStorageConfig


class EnergyStorage(Asset):
    CONFIG_CLASS = EnergyStorageConfig
    NAME = "energy storage"

    def step(
        self,
        control: list[DiscreteControl],
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> float:
        """Perform a step in the simulation, given a submitted control.

        Args:
            control (DiscreteControl): control to be performed.
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.
        """
        # We only need one control for this asset
        control_1 = control[0]

        # Get power associated with inputed control
        power = self.config.control_power_mapping[control_1.value]
        # If the internal energy is 0, we cannot deliver power
        if round(self.internal_energy) == 0 and power < 0:
            power = 0
        # If the internal energy is at max capacity, we cannot receive power
        elif round(self.internal_energy) == self.config.capacity_kwh and power > 0:
            power = 0
        
        self.power = power

        # Previous timestamp from last step
        previous_timestamp = self.timestamp

        # Datetime timestamp
        if isinstance(timestamp, dt.datetime) and isinstance(
            previous_timestamp, dt.datetime
        ):
            # Ensure time difference is non-zero at initialization
            if timestamp == previous_timestamp:
                time_difference = dt.timedelta(minutes=5)
            else:
                time_difference = timestamp - previous_timestamp
            energy_difference = power * time_difference.total_seconds() / 3600
        # Unknown or inconsistent timestamp type
        else:
            raise ValueError(f"timestamp type not supported: {type(timestamp)}.")

        # Update energy - must remain between 0 and max capacity
        self.internal_energy += energy_difference
        self.internal_energy = np.clip(self.internal_energy, 0, self.config.capacity_kwh)

        # Update state of charge
        self.state_of_charge = self.get_state_of_charge()

        # Log
        # log.debug(
        #    f"{timestamp} | asset {self.name:<{18}} | simulation step | "
        #    f"Power: {self.power:{9}.2f} kW | "
        #    f"Internal energy: {self.internal_energy:{9}.2f} kWh"
        # )

        # Run base class to store variables of interest
        super().step([control[0]], timestamp, outside_air_temperature)

        return self.power

    def get_auto_control(
        self,
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> list[DiscreteControl]:
        control_value = np.random.choice(list(self.config.control_power_mapping.keys()))

        control = [DiscreteControl(int(control_value))]
        return control

    def get_state_of_charge(self) -> float:
        """Get state of charge (SoC), in %, of the energy storage."""
        state_of_charge: float = (self.internal_energy / self.config.capacity_kwh) * 100
        return round(state_of_charge, 2)
