import datetime as dt

import pytest

from neuraflux.assets.energy_storage import (
    EnergyStorage,
)
from neuraflux.global_variables import POWER_KEY, TIMESTAMP_KEY
from neuraflux.schemas.asset_config import EnergyStorageConfig, EnergyStorageEfficiency
from neuraflux.schemas.control import DiscreteControl


# Fixture for reusable EnergyStorage instance
@pytest.fixture
def energy_storage():
    config = EnergyStorageConfig(initial_state_dict={"internal_energy": 250.0})
    timestamp = dt.datetime(2023, 11, 30, 17)
    return EnergyStorage("TestStorage", config, timestamp, 10)


def test_initialization(energy_storage):
    assert energy_storage.name == "TestStorage"
    assert energy_storage.config.capacity_kwh == 500.0
    # Add more assertions for other default values


def test_step_function(energy_storage):
    control = DiscreteControl(2)  # Choose a valid control value
    timestamp = dt.datetime(2023, 11, 30, 17, 5)
    power_output = energy_storage.step([control], timestamp, 5)
    assert power_output == energy_storage.config.control_power_mapping[control.value]
    # Add more assertions for internal state changes


def test_auto_step_function(energy_storage):
    # Save the initial state
    timestamp = dt.datetime(2023, 11, 30, 17, 5)

    # Perform the auto_step
    power_output = energy_storage.auto_step(timestamp, 5)

    # Assert that the power output is within the expected range
    assert (
        power_output in energy_storage.config.control_power_mapping.values()
    ), "Power output is not within the control power mapping."

    # Assert that internal state variables are updated
    assert energy_storage.timestamp == timestamp, "Timestamp was not updated correctly."

    # Check if history tracking was updated
    assert (
        power_output in energy_storage.history[POWER_KEY]
    ), "Power output not recorded in history."

    # Optionally, assert the length of the history to ensure it's growing
    history_length = len(energy_storage.history[TIMESTAMP_KEY])
    assert history_length > 0, "History did not update after auto_step."


def test_get_state_of_charge(energy_storage):
    # Manually set the internal energy
    test_energy = 250.0  # This is half of the default capacity of 500.0 kWh
    energy_storage.internal_energy = test_energy

    # Calculate expected state of charge
    expected_soc = (test_energy / energy_storage.config.capacity_kwh) * 100

    # Get the state of charge from the method
    calculated_soc = energy_storage.get_state_of_charge()

    # Assert if the calculated SoC matches the expected SoC
    assert calculated_soc == pytest.approx(
        expected_soc
    ), "The calculated state of charge is not as expected."


def test_efficiency_validation():
    with pytest.raises(ValueError):
        EnergyStorageEfficiency(
            value=1.1
        )  # Should raise ValueError as it's outside the valid range
