import datetime as dt
import os
import sqlite3
from unittest.mock import patch
import pandas as pd

import pytest
from neuraflux.geography import CityEnum
from neuraflux.global_variables import WEATHER_DB_NAME
from neuraflux.weather import Weather


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """Ensure the weather database file is removed after all tests are complete."""
    yield
    os.remove(WEATHER_DB_NAME) if os.path.exists(WEATHER_DB_NAME) else None


@pytest.fixture
def mock_city():
    """Fixture to provide a mock city instance."""
    return CityEnum.NEW_YORK


@pytest.fixture
def db_connection():
    """Provide an in-memory SQLite database connection and ensure it's closed after the test."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def weather_instance(mock_city):
    """Fixture to create a Weather instance with fixed dates for testing."""
    return Weather(mock_city, "", dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 3))


def test_weather_initialization(db_connection, mock_city):
    """Test the initialization of the Weather class."""
    with patch(
        "neuraflux.weather.create_connection_to_db", return_value=db_connection
    ) as mock_db_conn, patch(
        "neuraflux.weather.check_table_exists", return_value=False
    ) as mock_check_table, patch(
        "neuraflux.weather.preload_weather_data"
    ) as mock_preload:
        # Mock the call to read_sql to ensure the connection used is not closed
        with patch(
            "pandas.read_sql",
            return_value=pd.DataFrame(
                {"time": [dt.datetime(2022, 1, 1)], "temp": [22.0]}
            ),
        ):
            _ = Weather(mock_city, "", dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 2))

        mock_db_conn.assert_called_with(os.path.join("", WEATHER_DB_NAME))
        assert mock_db_conn.call_count == 2
        mock_check_table.assert_called_once()
        mock_preload.assert_called_once()


def test_get_weather_info_at_time(db_connection, weather_instance):
    """Test retrieval of weather information at a specified time."""
    # Setup the database with sample data
    db_connection.execute(
        f"CREATE TABLE {weather_instance.table_name} (time TEXT, temp REAL)"
    )
    sample_time = dt.datetime(2022, 1, 1, 1)
    db_connection.execute(
        f"INSERT INTO {weather_instance.table_name} VALUES (?, ?)",
        (sample_time.isoformat(), 22.0),
    )
    db_connection.commit()

    with patch("sqlite3.connect", return_value=db_connection):
        weather_info = weather_instance.get_weather_info_at_time(sample_time)
        assert weather_info is not None
        assert weather_info.temperature == 8.9
        # Test for a time with no data
        no_data_time = dt.datetime(2022, 1, 3, 1)
        with pytest.raises(ValueError):
            weather_instance.get_weather_info_at_time(no_data_time)
