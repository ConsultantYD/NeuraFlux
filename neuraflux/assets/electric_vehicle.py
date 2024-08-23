import datetime as dt
import logging as log

import numpy as np
import pandas as pd

from neuraflux.assets.base_asset import Asset
from neuraflux.schemas.control import DiscreteControl


class ElectricVehicle(Asset):
    def get_availability(self, timestamp: dt.datetime) -> int:
        np.random.seed(timestamp.day)
        morning_delta = int(np.random.randint(0, 18))
        afternoon_delta = int(np.random.randint(0, 18))

        new_timestamp = timestamp

        start_morning_commute = new_timestamp.replace(hour=6, minute=0) + dt.timedelta(
            minutes=morning_delta * 5
        )
        end_morning_commute = start_morning_commute + dt.timedelta(minutes=30)

        start_afternoon_commute = new_timestamp.replace(
            hour=15, minute=0
        ) + dt.timedelta(minutes=afternoon_delta * 5)
        end_afternoon_commute = start_afternoon_commute + dt.timedelta(minutes=30)

        # If current timestamp falls in any commute, set availability to 0
        if (
            start_morning_commute <= timestamp <= end_morning_commute
            or start_afternoon_commute <= timestamp <= end_afternoon_commute
        ):
            availability = 0
        else:
            availability = 1
        return availability

    def step(
        self,
        control: list[DiscreteControl],
        timestamp: dt.datetime,
        outside_air_temperature: float | None = None,
    ) -> float:
        """Perform a step in the simulation, given a submitted control.

        Args:
            control (DiscreteControl): control to be performed.
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.
        """
        # We only need one control for this asset
        control = control[0]

        # Random availability based on time of day
        self.availability = self.get_availability(timestamp)

        # Get power associated with inputed control
        self.power = self.config.control_power_mapping[control.value]
        self.power = 0 if self.power is None else self.power  # Linting

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
            energy_difference = self.power * time_difference.total_seconds() / 3600
        # Unknown or inconsistent timestamp type
        else:
            raise ValueError(f"timestamp type not supported: {type(timestamp)}.")

        # Update energy - must remain between 0 and max capacity

        # If not available, energy drops progressively and no power exchange with grid
        if self.availability == 0:
            self.power = 0
            energy_difference = -10

        self.internal_energy += energy_difference
        self.internal_energy = min(self.config.capacity_kwh, self.internal_energy)
        self.internal_energy = max(0, self.internal_energy)

        # Update state of charge
        self.state_of_charge = self.get_state_of_charge()

        # Log
        log.debug(
            f"{self.name} | Step: {timestamp} | "
            f"Power: {self.power} W. | "
            f"Internal energy: {self.internal_energy} kWh."
        )

        # Run base class to store variables of interest
        super().step([control], timestamp, outside_air_temperature)

        return self.power

    def get_auto_control(
        self,
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> list[DiscreteControl]:
        # control_value = np.random.choice(
        #    list(self.config.control_power_mapping.keys())
        # )
        control_value = 2 if self.internal_energy < self.config.capacity_kwh else 1
        control = [DiscreteControl(int(control_value))]
        return control

    def get_state_of_charge(self) -> float:
        """Get state of charge (SoC), in %, of the energy storage."""
        state_of_charge: float = (self.internal_energy / self.config.capacity_kwh) * 100
        return round(state_of_charge, 2)

    def augment_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Loop over all df rows
        availabilities = []
        for index, row in df.iterrows():
            # Get timestamp
            timestamp = row.name
            availabilities.append(self.get_availability(timestamp))
        df["availability"] = availabilities
        return df
