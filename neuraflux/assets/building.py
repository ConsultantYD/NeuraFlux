import datetime as dt

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neuraflux.assets.base_asset import Asset
from neuraflux.schemas.asset_config import BuildingConfig
from neuraflux.schemas.control import ContinuousControl, DiscreteControl


class Building(Asset):
    def __init__(
        self,
        name: str,
        config: BuildingConfig,
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ):
        dt = config.dt
        self.config = config

        self.Uinv, self.F, self.C = get_simulation_properties_for_building(dt)

        self.n_zones = self.Uinv.shape[0]
        self.occupancy = self.get_occupancy_value(timestamp)
        self.heat_setpoint, self.cool_setpoint = self.get_setpoint_values(timestamp)

        super().__init__(name, config, timestamp, outside_air_temperature)

    def step(
        self,
        control: list[DiscreteControl],
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> float:
        # Store outside air temperature for this step
        self.outside_air_temperature = outside_air_temperature

        # Define the new state of the HVAC system based on controls
        self.hvac = [c.value - 2 for c in control]

        # Get power associated with each control and define global power
        power_vec = [
            self.config.control_power_mapping[c.value]
            if c.value in (2, 3, 4)
            else -self.config.control_power_mapping[c.value]
            for c in control
        ]
        self.power = float(np.sum(np.abs(power_vec)))

        self.temperature = self._update_room_temperature(power_vec)

        self.heat_setpoint, self.cool_setpoint = self.get_setpoint_values(timestamp)

        # Define occupancy value
        self.occupancy = self.get_occupancy_value(timestamp)

        # Run base class to store variables of interest
        super().step(control, timestamp, outside_air_temperature)
        return self.power

    def get_auto_control(
        self, timestamp: dt.datetime, outside_air_temperature: float
    ) -> list[DiscreteControl]:
        setpoints = self.get_setpoint_values(timestamp)
        control = []
        for hvac, temp in zip(self.hvac, self.temperature):
            # If temperature is too hot ...
            if temp > setpoints[1]:
                # Keep stage 2 on if it was already on
                if hvac == -2:
                    control.append(ContinuousControl(0))
                # Stage 2 cooling if gap is bigger than 1 degree
                elif temp - setpoints[1] > 1:
                    control.append(ContinuousControl(0))
                # Stage 1 cooling otherwise
                else:
                    control.append(ContinuousControl(1))
            # If temperature is too cold ...
            elif temp < setpoints[0]:
                # Keep stage 2 on if it was already on
                if hvac == 2:
                    control.append(ContinuousControl(4))
                # Stage 2 heating if gap is bigger than 1 degree
                elif setpoints[0] - temp > 1:
                    control.append(ContinuousControl(4))
                # Stage 1 heating otherwise
                else:
                    control.append(ContinuousControl(3))
            # If temperature is within setpoints ...
            else:
                control.append(ContinuousControl(2))
        return control

    def augment_df(self, df) -> pd.DataFrame:
        df = df.copy()

        # Loop over all indexes of the dataframe
        prev_idx = None
        for idx in df.index:
            heat_setpoint, cool_setpoint = self.get_setpoint_values(idx)
            occupancy = self.get_occupancy_value(idx)

            # Add setpoints and occupancy
            if np.isnan(df.loc[idx, "heat_setpoint"]):
                df.loc[idx, "heat_setpoint"] = heat_setpoint
            if np.isnan(df.loc[idx, "cool_setpoint"]):
                df.loc[idx, "cool_setpoint"] = cool_setpoint
            if np.isnan(df.loc[idx, "occupancy"]):
                df.loc[idx, "occupancy"] = occupancy

            # Add control values
            if prev_idx is not None and df.loc[prev_idx, "control_1"] is not None:
                df.loc[idx, "hvac_1"] = df.loc[idx, "control_1"] - 2
            if prev_idx is not None and df.loc[prev_idx, "control_2"] is not None:
                df.loc[idx, "hvac_2"] = df.loc[idx, "control_2"] - 2
            if prev_idx is not None and df.loc[prev_idx, "control_3"] is not None:
                df.loc[idx, "hvac_3"] = df.loc[idx, "control_3"] - 2
            prev_idx = idx
        return df

    def get_occupancy_value(self, time) -> bool:
        # Occupied setpoints
        if (
            time.hour >= self.config.occ_times[0]
            and time.hour < self.config.occ_times[1]
        ):
            return 1.0
        # Unoccupied setpoints
        return 0.0

    def get_setpoint_values(self, time) -> tuple[float, float]:
        is_occupied = self.get_occupancy_value(time)
        # Occupied setpoints
        if is_occupied:
            return self.config.occ_setpoints
        # Unoccupied setpoints
        return self.config.unocc_setpoints

    def _update_room_temperature(self, power_vec: list[float]):
        # HVAC Q input (in Watts)
        Q_hvac: NDArray[np.float_] = np.array(power_vec) * 1000

        # Outside heat input (e.g. Solar)
        Q_in = np.zeros(len(self.Uinv))  # None for now

        # Get total Q contribution (include HVAC by default)
        term1 = np.dot(self.F, self.outside_air_temperature)
        term2 = np.multiply(
            self.C.T / (self.config.dt * 60), self.temperature
        ).flatten()
        Q = Q_hvac + Q_in + term1 + term2

        # Calculate new temperature
        return np.dot(Q, self.Uinv).diagonal().tolist()


# Thermal circuit model constructor
def mUxFxCx(
    Uin: NDArray[np.float_],
    F: NDArray[np.float_],
    C: NDArray[np.float_],
    dt: int,
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    """Generalized multi-zone RC model constructor. nN is the number of nodes,
        nM is the number of nodes with known temperatures / boundaries

    Args:
        Uin (np.ndarray): Conductance matrix input by user, upper triangle only, (nN x nN)(W / K).
        F (np.ndarray): Conductance matrix of nodes connected to a known temperature source, (nN x nM)(W / K).
        C (np.ndarray): Capacitance vector, (nN x 1)(J / K).
        dt (int): Time step used for the simulation.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: Inverse of U, Conductance matrix of nodes connected to a known temperature source (nN x nM)(W / K) and Capacitance vector (nN x 1)(J / K).
    """  # noqa E501

    # Number of internal nodes
    nN = len(Uin)

    # U-matrix completion, and its inverse
    U: NDArray[np.float_] = -Uin - Uin.T  # U is symmetrical, non-diagonals are -ve
    s: NDArray[np.float_] = -np.sum(U, 1)
    for i in range(0, nN):
        U[i, i] = s[i] + np.sum(F[i, :]) + C[i][0] / dt
    U_inv = np.linalg.inv(U)

    # Function return | Matrices needed
    return U_inv, F, C


# Thermal circuits bank
def get_simulation_properties_for_building(
    dt: int = 5,
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    """Creates the following representation:

            +----------++----------++----------+
            |          ||          ||          |
            |  Zone 0  ||  Zone 1  ||  Zone 3  |
    T_ext   |          ||          ||          | T_ext (=F0)
         F00|          || U01      || U02      | F10
       O---www---o----wwww----o---wwww---o----www---O
            |    |C0   ||     |C1  |     |C2   |
            |   ---    ||    ---   |    ---    |
            |   ---    ||    ---   |    ---    |
            |    |     ||     |    |     |     |
            |   GND    ||    GND   |    GND    |
            +----------++----------+-+---------+

    Args:
        dt (int, optional): Simulation time step. Defaults to 5.

    Returns:
        tuple: RC model constants for this building type.
    """

    # Nodal Connections
    nN = 3  # number of nodes: zone 0, 1 and 2
    nM = 1  # number of nodes with known temperatures: T_exterior

    # Declare variables
    Uin = np.zeros((nN, nN))  # W/K
    F = np.zeros((nN, nM))  # W/K
    C = np.zeros((nN, 1))  # J/K

    # How are the nodes connected (admittance) ?
    Uin[0, 1] = 800.0  # Wall between Zone 1 and 2
    Uin[0, 2] = 1000.0  # Wall between Zone 1 and 2

    # Connected to temperature sources
    F[0, 0] = 300.0  # Zone 0 towards exterior
    F[1, 0] = 50  # Zone 2 towards exterior
    F[2, 0] = 50  # Zone 3 towards exterior

    # Nodes with capacitance
    C[0] = 11e6  # Zone 0 thermal capacitance
    C[1] = 11e6  # Zone 1 thermal capacitance
    C[2] = 10e6  # Zone 2 thermal capacitance

    # Function return | Matrices from RC model constructor
    return mUxFxCx(Uin, F, C, dt * 60)
