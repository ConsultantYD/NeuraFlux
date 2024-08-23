import datetime as dt
import os
from dataclasses import dataclass

import pandas as pd
from meteostat import Hourly, Point

from neuraflux.geography import CityEnum
from neuraflux.global_variables import CITIES_INFO, WEATHER_DB_NAME
from neuraflux.utils_sql import (
    add_dataframe_to_table,
    check_table_exists,
    create_connection_to_db,
)


@dataclass
class WeatherInfo:
    temperature: float


class Weather:
    def __init__(
        self,
        city: CityEnum,
        db_dir: str = "",
        start_date=dt.datetime(2020, 1, 1, 0, 0, 0),
        end_date=dt.datetime(2024, 1, 1, 0, 0, 0),
    ):
        """Initialize the Weather instance with a city."""
        self.city = city
        self.table_name = f"{city.lower().replace(' ', '_')}_weather"
        self.db_path = os.path.join(db_dir, WEATHER_DB_NAME)

        # Check if data needs to be preloaded
        conn = create_connection_to_db(self.db_path)
        if not check_table_exists(conn, self.table_name):
            preload_weather_data(
                **CITIES_INFO[city],
                table_name=self.table_name,
                conn=conn,
                start_date=start_date,
                end_date=end_date,
            )
        conn.close()

        # Load weather data in memory
        self.weather_data = pd.read_sql(
            f"SELECT * FROM {self.table_name}",  # nosec
            create_connection_to_db(self.db_path),
        )
        # Set time as datetime and index
        self.weather_data["time"] = pd.to_datetime(self.weather_data["time"])

    def get_weather_info_at_time(self, time: dt.datetime) -> WeatherInfo:
        temperature = self.get_temperature_at_time(time)
        return WeatherInfo(temperature=temperature) if temperature is not None else None

    def get_temperature_at_time(self, time: dt.datetime) -> float:
        temp_vec = self.weather_data.loc[
            self.weather_data["time"] == time, "temp"
        ].values
        match len(temp_vec):
            case 0:
                raise ValueError(
                    "No temperature data available for the specified time."
                )
            case 1:
                temp = temp_vec[0]
            case _:
                raise ValueError("More than one temperature value found.")
        return temp


def preload_weather_data(
    lat,
    lon,
    alt,
    table_name,
    conn,
    start_date=dt.datetime(2020, 1, 1, 0, 0, 0),
    end_date=dt.datetime(2024, 1, 1, 0, 0, 0),
):
    """Preload weather data for a city."""

    # Fetch weather data
    location = Point(lat, lon, alt)
    retriever = Hourly(location, start_date, end_date)
    weather_data = retriever.fetch()
    weather_data.reset_index(inplace=True)  # Reset index to make 'time' a column

    # Interpolate to minute-level granularity
    weather_data.set_index("time", inplace=True)
    weather_data = weather_data.resample("1T").interpolate(method="linear")
    weather_data.reset_index(inplace=True)

    # Create table if it doesn't exist and insert data
    add_dataframe_to_table(weather_data, conn, table_name, "time")
