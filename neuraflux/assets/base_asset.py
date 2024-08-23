import datetime as dt
import logging as log
from abc import ABCMeta, abstractmethod
from copy import copy
from os.path import join
from typing import Any, Optional, Union

import dill
import numpy as np
import pandas as pd
from neuraflux.global_variables import (
    CONTROL_KEY,
    LOG_ENTITY_KEY,
    LOG_MESSAGE_KEY,
    LOG_METHOD_KEY,
    LOG_SIM_T_KEY,
    TIMESTAMP_KEY,
)
from neuraflux.schemas.control import DiscreteControl


class Asset(metaclass=ABCMeta):
    """Base class for all assets."""

    def __init__(
        self,
        name: str,
        config: Any,
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> None:
        # INITIALIZE INTERNAL ASSET VARIABLES
        # Basic asset information
        self.name = name
        self.config = config

        # Core variables
        self.timestamp: dt.datetime = timestamp
        self.outside_air_temperature: float = outside_air_temperature
        self.control: list[int] = []
        self.power: float | None = None

        # TRACKED VARIABLE AND HISTORY
        # Initialize history for variables tracked in the simulation
        tracked_variables = self._get_tracked_variables()

        # Initialize variables history, with care for array-like inputs
        self.history = self._initialize_history(
            tracked_variables, self.config.initial_state_dict
        )

        # Use inputed initial values as initial state
        for key, value in self.config.initial_state_dict.items():
            setattr(self, key, value)
        self._update_tracked_variables()

        # Logging
        asset_class = self.__class__.__name__
        log.debug(
            {
                LOG_SIM_T_KEY: self.timestamp,
                LOG_ENTITY_KEY: f"Asset({self.name}) - {asset_class}",
                LOG_METHOD_KEY: "__init__",
                LOG_MESSAGE_KEY: f"Initializing asset with: "
                f"{', '.join([f'{var}: {getattr(self, var) if isinstance(getattr(self, var), (int, float)) else str(getattr(self, var)):5.2f}' if isinstance(getattr(self, var), (int, float)) else f'{var}: {str(getattr(self, var))}' for var in self.config.initial_state_dict.keys()])}",
            }
        )

    @abstractmethod
    def step(
        self,
        control: list[DiscreteControl],
        timestamp: int | dt.datetime,
        outside_air_temperature: float,
    ) -> Optional[float]:
        """Perform a step in the simulation, given a submitted control.
        This function must return the power consumption of the asset at
        this step step. Positive values indicate power consumption, while
        negative values indicate power generation.

        Args:
            control (int): control to be performed.
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.
            outside_air_temperature (float): outside air temperature.

        Returns:
            float: Power consumed by the asset.
        """
        # Update core variables
        self.timestamp = timestamp
        self.control = [c.value for c in control]
        self.outside_air_temperature = outside_air_temperature
        self._update_tracked_variables()

        # Logging
        tracked_variables = self.config.tracked_variables
        asset_class = self.__class__.__name__
        log.debug(
            {
                LOG_SIM_T_KEY: self.timestamp,
                LOG_ENTITY_KEY: f"Asset({self.name}) - {asset_class}",
                LOG_METHOD_KEY: "step",
                LOG_MESSAGE_KEY: f"controls {str([c.value for c in control])} ({self.power:.2f} kW) modified asset variables to: "
                f"{'| '.join([f'{var}: {getattr(self, var):5.2f}' if isinstance(getattr(self, var), (int, float)) else f'{var}: {str(getattr(self, var))}' for var in tracked_variables])}",
            }
        )

        # Convert array-like signals to individual variables
        for var in self.config.tracked_variables:
            if isinstance(getattr(self, var), (list, tuple, np.ndarray)):
                for i in range(len(getattr(self, var))):
                    setattr(self, var + f"_{i+1}", getattr(self, var)[i])

        # NOTE: The child class must define the power variable
        return self.power

    @abstractmethod
    def get_auto_control(
        self, timestamp: Union[int, dt.datetime], outside_air_temperature: float
    ) -> list[DiscreteControl]:
        """Get the automatic control for the asset.

        Args:
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.
            outside_air_temperature (float): outside air temperature.

        Returns:
            list[DiscreteControl]: The automatic control.
        """
        raise NotImplementedError

    def auto_step(
        self,
        timestamp: Union[int, dt.datetime],
        outside_air_temperature: float | None,
    ) -> float:
        """Perform a step in the simulation, automatically choosing control.

        Args:
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.
            outside_air_temperature (float): outside air temperature.

        Returns:
            float: Power consumed by the asset.
        """
        control = self.get_auto_control(timestamp, outside_air_temperature)
        return self.step(control, timestamp, outside_air_temperature)

    def augment_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_signal(self, signal: str) -> Any:
        if hasattr(self, signal):
            return getattr(self, signal)

        return np.nan

    def get_historical_data(self, nan_padding: bool = False) -> pd.DataFrame:
        """Returns a dataframe with all variables tracked in the simulation.

        Returns:
            pd.DataFrame: Dataframe with all variables tracked in simulation.
        """

        # Work with a copy of the data to avoid modifying the original
        data = copy(self.history)

        # Add None values to the end of the list to equalize length
        if nan_padding:
            max_length = max(len(value) for value in data.values())
            data_clean = {
                key: value + [None] * (max_length - len(value))
                for key, value in data.items()
            }

        # Remove last timestamps missing control and power data
        else:
            min_length = min(len(value) for value in data.values())
            data_clean = {key: value[:min_length] for key, value in data.items()}

        # Separate index column and clean data columns to create dataframe
        timestamp_data = data_clean.pop(TIMESTAMP_KEY)
        df = pd.DataFrame(data_clean, index=timestamp_data)
        return df

    def to_file(self, directory: str = "") -> None:
        """
        Save asset's state to a file using dill.

        Args:
            filename (str, optional): Name of the file to save.
            directory (str, optional): Save directory. Defaults to "".
        """
        filepath = join(directory, self.name + ".pkl")
        with open(filepath, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def from_file(cls, name: str, directory: str = "") -> "Asset":
        """
        Load an asset's state from a file using dill.

        Args:
            filename (str): Name of the file to load from.
            directory (str, optional): Load directory. Defaults to "".
        """
        filepath = join(directory, name + ".pkl")
        with open(filepath, "rb") as f:
            return dill.load(f)

    def _get_tracked_variables(self) -> list[str]:
        core_variables = self.config.core_variables
        asset_specific_variables = self.config.tracked_variables
        all_tracked_variables = core_variables + asset_specific_variables

        # Handle control separately, as it is a list of unknown size
        all_tracked_variables.remove(CONTROL_KEY)
        all_tracked_variables += [
            CONTROL_KEY + "_" + str(i + 1) for i in range(self.config.n_controls)
        ]

        return all_tracked_variables

    def _initialize_history(
        self, tracked_variables: list[str], initial_state_dict: [str, Any]
    ) -> dict[str, list]:
        history = {}
        init_variables = initial_state_dict
        for variable in tracked_variables:
            if variable in init_variables and isinstance(
                init_variables[variable], (list, tuple, np.ndarray)
            ):
                for i in range(len(init_variables[variable])):
                    history[variable + "_" + str(i + 1)] = []
            else:
                history[variable] = []
        return history

    def _update_tracked_variables(self) -> None:
        # Define all tracked variables to add to history
        tracked_variables = self.config.tracked_variables + self.config.core_variables

        # Add all other standard variables
        for variable_name in tracked_variables:
            variable = getattr(self, variable_name)
            if isinstance(variable, (list, tuple, np.ndarray)):
                for i, value in enumerate(variable):
                    self.history[variable_name + "_" + str(i + 1)].append(value)
            elif variable is not None:
                self.history[variable_name].append(variable)
