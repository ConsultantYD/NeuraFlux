import pytest
import datetime as dt
from neuraflux.assets.building import Building, BuildingConfig
from neuraflux.schemas.control import DiscreteControl


# Fixture for reusable Building instance
@pytest.fixture
def building():
    config = BuildingConfig()
    timestamp = dt.datetime(2023, 11, 30, 17)
    outside_air_temp = 5
    return Building("TestBuilding", config, timestamp, outside_air_temp)


def test_initialization(building):
    assert building.name == "TestBuilding"
    assert building.config.n_controls == 3
    # Add more assertions for other default values like temperature, hvac, etc.
    assert building.temperature == [21.0, 21.0, 21.0]
    assert building.hvac == [0, 0, 0]


def test_step_function(building):
    controls = [DiscreteControl(2), DiscreteControl(3), DiscreteControl(4)]
    timestamp = dt.datetime(2023, 11, 30, 17)
    outside_air_temp = 6
    power_output = building.step(controls, timestamp, outside_air_temp)
    assert building.timestamp == timestamp
    assert building.outside_air_temperature == outside_air_temp
    assert power_output == sum(
        [building.config.control_power_mapping[c.value] for c in controls]
    )


def test_building_get_historical_data(building):
    controls = [DiscreteControl(4), DiscreteControl(4), DiscreteControl(4)]
    timestamp = dt.datetime(2023, 11, 30, 17)
    outside_air_temp = 6
    for _ in range(3):
        building.step(controls, timestamp, outside_air_temp)
        timestamp = timestamp + dt.timedelta(minutes=5)
        controls = [DiscreteControl(0), DiscreteControl(0), DiscreteControl(0)]
    _ = building.get_historical_data()


def test_auto_step_function(building):
    timestamp = dt.datetime(2023, 11, 30, 17)
    for _ in range(10):
        building.auto_step(timestamp, -10.0)
        assert building.timestamp == timestamp
        timestamp = timestamp + dt.timedelta(minutes=5)


def test_temperature_calculation(building):
    controls = [DiscreteControl(0), DiscreteControl(0), DiscreteControl(0)]
    timestamp = dt.datetime(2023, 11, 30, 17)
    outside_air_temp = 25.0
    building.step(controls, timestamp, outside_air_temp)
    # Add assertions to verify temperature calculations are correct


def test_history_tracking(building):
    controls = [DiscreteControl(2)]
    timestamp = dt.datetime(2023, 11, 30, 17)
    outside_air_temp = 25.0
    building.step(controls, timestamp, outside_air_temp)
    assert timestamp in building.history["timestamp"]
    assert building.power in building.history["power"]


def test_exception_handling(building):
    # Test with invalid controls or timestamps
    with pytest.raises(ValueError):
        invalid_controls = [
            DiscreteControl(-1)
        ]  # Assuming -1 is an invalid control value
        timestamp = dt.datetime(2023, 11, 30, 17)
        building.step(invalid_controls, timestamp, 25.0)
