import datetime as dt
import os
from typing import Any

import dill
import pandas as pd

from neuraflux.time_ref import convert_datetime_to_unix
from neuraflux.utils_sql import (
    add_dataframe_to_table,
    create_connection_to_db,
)


class Module:
    """
    A base class for all modules in the NeuraFlux framework.

    Methods:
        to_file: Saves the module to a file.
        from_file: Loads the module from a file.
    """

    def __init__(self, base_dir: str = ""):
        self.base_dir = base_dir

    def to_file(self, directory: str = "") -> None:
        """
        Saves the module to a file.

        Args:
            directory (str): The directory to save the file in.

        Returns:
            None
        """
        filepath = os.path.join(directory, self.__class__.__name__)
        with open(filepath, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def from_file(cls, directory: str = "") -> "Module":
        filepath = os.path.join(directory, cls.__name__)
        try:
            with open(filepath, "rb") as f:
                instance = dill.load(f)
        except FileNotFoundError:
            instance = None
        return instance

    @classmethod
    def load_or_initialize(cls, directory: str = ""):
        instance = cls.from_file(directory)
        if instance is None:
            instance = cls()
        return instance

    def store_data_in_table_at_time(
        self,
        uid: str,
        table_name: str,
        timestamp: dt.datetime,
        data: dict[str, Any],
        data_columns: list[str] | None = None,
    ):
        db_connection = self.create_connection_to_agent_db(uid)
        # Convert timestamp to unix timestamp (assumes UTC timezone)
        unix_timestamp = convert_datetime_to_unix(timestamp)

        # Put signals as DataFrame
        df = pd.DataFrame([{"wavecounter": unix_timestamp, "uid": uid, **data}])

        # Ensure columns are always there and in the same order
        if data_columns is not None:
            data_columns = ["wavecounter", "uid"] + data_columns
            df = df[data_columns]

        # Store the signal data in the database
        add_dataframe_to_table(df, db_connection, table_name, index_col="wavecounter")

        # Delete unused variables and force garbage collection
        del df, data, data_columns

    def get_data_from_table(
        self,
        uid: str,
        table_name: str,
        start_time: dt.datetime = None,
        end_time: dt.datetime = None,
        columns: list = None,
    ):
        """
        Retrieve data from a table for a given UID within an optional datetime range and specific columns.

        Parameters:
        -----------
        uid : str
            The unique identifier for the asset.
        table_name : str
            The name of the table to retrieve data from.
        start_dt : datetime.datetime, optional
            The start datetime for filtering the records.
        end_dt : datetime.datetime, optional
            The end datetime for filtering the records.
        columns : list of str, optional
            Specific columns to retrieve; retrieves all columns if None.

        Returns:
        --------
        DataFrame
            A DataFrame containing the retrieved asset signals.
        """
        db_connection = self.create_connection_to_agent_db(uid)
        # General query to retrieve everything
        query = f"SELECT * FROM {table_name} WHERE uid = '{uid}'"  # nosec

        # If start time is provided ...
        if start_time is not None:
            start_unix = convert_datetime_to_unix(start_time)
            query += f" AND wavecounter >= {start_unix}"

        # If end time is provided ...
        if end_time is not None:
            end_unix = convert_datetime_to_unix(end_time)
            query += f" AND wavecounter <= {end_unix}"

        # Perform query and define dataframe
        df = pd.read_sql_query(query, db_connection)
        if columns is not None:
            df = df[columns]

        # Add datetime column
        if "wavecounter" in df.columns:
            df["timestamp"] = pd.to_datetime(df["wavecounter"], unit="s", utc=True)
            df.set_index("timestamp", inplace=True)

            # Remove uid and wavecounter columns
            df.drop(columns=["wavecounter", "uid"], inplace=True)

        return df

    # Create database file for time series data
    def create_connection_to_agent_db(self, uid: str) -> Any:
        # Create agent directory
        agent_dir = os.path.join(self.base_dir, uid)

        # Create database connection
        db_file = os.path.join(agent_dir, f"{uid}.db")
        db_connection = create_connection_to_db(db_file)
        return db_connection
