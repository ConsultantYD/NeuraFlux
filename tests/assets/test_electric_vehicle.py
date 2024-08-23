import datetime as dt

import pytest

from neuraflux.assets.electric_vehicle import ElectricVehicle
from neuraflux.global_variables import POWER_KEY, TIMESTAMP_KEY
from neuraflux.schemas.asset_config import (
    EnergyStorageConfig,
)
#from neuraflux.schemas.control import DiscreteControl


# Fixture for reusable ElectricVehicle instance
@pytest.fixture
def electric_vehicle():
    config = EnergyStorageConfig(initial_state_dict={"internal_energy": 50.0})
    timestamp = dt.datetime(2023, 11, 30, 17)
    return ElectricVehicle("TestEV", config, timestamp, 10)


def test_initialization(electric_vehicle):
    assert electric_vehicle.name == "TestEV"
    assert electric_vehicle.config.capacity_kwh == 500.0
    # Add more assertions for other default values


def test_auto_step_function(electric_vehicle):
    # Save the initial state
    timestamp = dt.datetime(2023, 11, 30, 17, 5)

    # Perform the auto_step
    power_output = electric_vehicle.auto_step(timestamp, 5)

    # Assert that the power output is within the expected range
    assert (
        power_output in electric_vehicle.config.control_power_mapping.values()
    ), "Power output is not within the control power mapping."

    # Assert that internal state variables are updated
    assert (
        electric_vehicle.timestamp == timestamp
    ), "Timestamp was not updated correctly."

    # Check if history tracking was updated
    assert (
        power_output in electric_vehicle.history[POWER_KEY]
    ), "Power output not recorded in history."

    # Optionally, assert the length of the history to ensure it's growing
    history_length = len(electric_vehicle.history[TIMESTAMP_KEY])
    assert history_length > 0, "History did not update after auto_step."


def test_get_state_of_charge(electric_vehicle):
    # Manually set the internal energy
    test_energy = 250.0  # This is half of the default capacity of 500.0 kWh
    electric_vehicle.internal_energy = test_energy

    # Calculate expected state of charge
    expected_soc = (test_energy / electric_vehicle.config.capacity_kwh) * 100

    # Get the state of charge from the method
    calculated_soc = electric_vehicle.get_state_of_charge()

    # Assert if the calculated SoC matches the expected SoC
    assert calculated_soc == pytest.approx(
        expected_soc
    ), "The calculated state of charge is not as expected."
