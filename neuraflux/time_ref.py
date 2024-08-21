import datetime as dt
import json
import os
from dataclasses import dataclass

import pytz

from neuraflux.global_variables import DT_STR_FORMAT


@dataclass(frozen=True)
class TimeInfo:
    t: dt.datetime
    dt: dt.timedelta

    def get_previous_TimeInfo(self) -> "TimeInfo":
        t = self.t - self.dt
        dt = self.dt
        return TimeInfo(t=t, dt=dt)

    def get_next_TimeInfo(self) -> "TimeInfo":
        t = self.t + self.dt
        dt = self.dt
        return TimeInfo(t=t, dt=dt)

    def get_t_as_str(self) -> str:
        return self.t.strftime(DT_STR_FORMAT)


class TimeRef:
    def __init__(
        self,
        start_time_utc: dt.datetime,
        def_time_step: dt.timedelta = dt.timedelta(minutes=5),
    ) -> None:
        """Initialise the time module.

        Args:
            time_utc (dt.datetime): The initial time, in UTC.
            directory (str, optional): The directory to save the time
        """
        self.time_utc = start_time_utc
        self.initial_time_utc = start_time_utc
        self.def_time_step = def_time_step

    def get_time_info(self) -> TimeInfo:
        return TimeInfo(t=self.get_time_utc(), dt=self.def_time_step)

    def get_time_utc(self) -> dt.datetime:
        """Get the current time, in UTC.

        Returns:
            dt.datetime: Current time, in UTC
        """
        return self.time_utc

    def get_time_utc_as_str(self) -> str:
        """Get the current time, in UTC, as a string.

        Returns:
            str: Current time, in UTC, as a string
        """
        return self.time_utc.strftime(DT_STR_FORMAT)

    def get_initial_time_utc(self) -> dt.datetime:
        """Get the starting time, in UTC.

        Returns:
            dt.datetime: Starting time, in UTC
        """
        return self.initial_time_utc

    def get_initial_time_utc_as_str(self) -> str:
        """Get the starting time, in UTC, as a string.

        Returns:
            str: Starting time, in UTC, as a string
        """
        return self.initial_time_utc.strftime(DT_STR_FORMAT)

    def increment_time(self, delta: dt.timedelta | int = None) -> None:
        """Increment the current time by the specified delta.

        Args:
            delta (Union[dt.timedelta, int]): The delta to increment
            the time by. Integers will be converted to seconds.
        """
        if delta is None:
            delta = self.def_time_step
        if isinstance(delta, (int, float)):
            delta = dt.timedelta(seconds=delta)
        self.time_utc += delta

    def to_file(self, directory: str) -> None:
        file_path = os.path.join(directory, "time_ref.json")
        time_dict = {
            "time_utc": self.time_utc.strftime(DT_STR_FORMAT),
            "initial_time_utc": self.initial_time_utc.strftime(DT_STR_FORMAT),
        }
        with open(file_path, "w") as f:
            json.dump(time_dict, f, indent=4)

    def get_time_delta_since_start(self) -> dt.timedelta:
        """Get the time delta since the initial time.

        Returns:
            dt.timedelta: The time delta since the initial time.
        """
        return self.time_utc - self.initial_time_utc

    @classmethod
    def from_file(cls, directory: str) -> "TimeRef":
        file_path = os.path.join(directory, "time_ref.json")
        with open(file_path, "r") as f:
            time_dict = json.load(f)
        time_utc = dt.datetime.strptime(time_dict["time_utc"], DT_STR_FORMAT)
        initial_time_utc = dt.datetime.strptime(
            time_dict["initial_time_utc"], DT_STR_FORMAT
        )
        self = cls(start_time_utc=time_utc)
        self.initial_time_utc = initial_time_utc
        return self


def convert_datetime_to_unix(timestamp: dt.datetime) -> int:
    """Convert a datetime object to a unix timestamp.

    Args:
        timestamp (dt.datetime): The datetime object to convert.

    Returns:
        int: The unix timestamp.
    """
    # Make sure datetime is UTC aware
    timestamp = timestamp.replace(tzinfo=pytz.utc)

    # Perform conversion
    unix_timestamp = int(timestamp.timestamp())
    return unix_timestamp
