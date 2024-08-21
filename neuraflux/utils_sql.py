import json
import logging as log
import sqlite3
from sqlite3 import DatabaseError

import pandas as pd


def create_connection_to_db(db_file: str, memory: bool = False) -> sqlite3.Connection:
    """Create a database connection to a SQLite database.

    Args:
    db_file (str): The file path of the database file.
    memory (bool): If True, a temporary in-memory database is created.

    Returns:
    sqlite3.Connection: The connection object to the SQLite database.

    Raises:
    sqlite3.DatabaseError: If an error occurs while attempting to connect to the database.
    """
    conn = None
    db_file = ":memory:" if memory else db_file
    try:
        conn = sqlite3.connect(db_file)
    except DatabaseError as e:
        log.error(f"Failed to connect to the database: {e}")
        raise  # Propagate the error after logging it

    return conn


def infer_sqlite_type(dtype) -> str:
    """Map pandas dtype to SQLite data type."""
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    else:
        return "TEXT"


def check_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the SQLite database.

    Args:
    conn (sqlite3.Connection): Connection object to the SQLite database.
    table_name (str): Name of the table to check.

    Returns:
    bool: True if the table exists, False otherwise.
    """
    check_query = (
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"  # nosec
    )
    cur = conn.cursor()
    cur.execute(check_query)
    return cur.fetchone() is not None


# def create_table_from_dataframe(
#     df: pd.DataFrame, conn: sqlite3.Connection, table_name: str
# ) -> None:
#     """Create a SQL table from a pandas DataFrame schema in a SQLite database.

#     Args:
#     df (pd.DataFrame): The DataFrame from which to infer the schema.
#     conn (sqlite3.Connection): Connection object to the SQLite database.
#     table_name (str): Name of the table to be created.
#     """

#     try:
#         cur = conn.cursor()
#         # Remove dataframe data and keep only the schema
#         df_empty = df.iloc[0:0]
#         # Create the table with the schema inferred from the DataFrame
#         df_empty.to_sql(table_name, conn, if_exists="replace", index=False)
#         log.debug(f"Table '{table_name}' created successfully.")
#     except DatabaseError as e:
#         log.error(f"An error occurred while creating the table: {e}")
#         raise  # Propagate the error after logging it


def add_dataframe_to_table(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    table_name: str,
    index_col: str | None = None,
    use_index: bool = True,
) -> None:
    if index_col is not None:
        df = df.copy().set_index(index_col)

    # Append the updated DataFrame to the SQL table
    df.to_sql(table_name, conn, if_exists="append", index=use_index)

    # Clear the DataFrame from memory
    df = None


####################################################################################################
# KEY-VALUE STORAGE
####################################################################################################
def create_key_value_table(conn: sqlite3.Connection, table_name: str) -> None:
    """Create a key-value table in the SQLite database if it does not exist.

    Args:
    conn (sqlite3.Connection): Connection object to the SQLite database.
    table_name (str): Name of the key-value table to be created. Default is "key_value_store".
    """
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            key TEXT PRIMARY KEY,
            value BLOB
        );
    """
    try:
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
    except DatabaseError as e:
        log.error(f"An error occurred while creating the key-value table: {e}")
        raise  # Propagate the error after logging it


def store_value_in_db(
    conn: sqlite3.Connection, key: str, value, table_name: str
) -> None:
    """Store a value in a key-value table in the SQLite database.

    Args:
    conn (sqlite3.Connection): Connection object to the SQLite database.
    key (str): Key under which the value should be stored.
    value: Data to be stored, which will be serialized to JSON.
    table_name (str): Name of the key-value table.
    """
    serialized_value = json.dumps(value)
    insert_query = f"INSERT OR REPLACE INTO {table_name} (key, value) VALUES (?, ?);"
    try:
        cur = conn.cursor()
        cur.execute(insert_query, (key, serialized_value))
        conn.commit()
    except DatabaseError as e:
        log.error(f"Failed to store data under key '{key}': {e}")
        raise  # Propagate the error after logging it


def retrieve_value_from_db(conn: sqlite3.Connection, key: str, table_name: str):
    """Retrieve a value from a key-value table in the SQLite database.

    Args:
    conn (sqlite3.Connection): Connection object to the SQLite database.
    key (str): Key whose associated value is to be retrieved.
    table_name (str): Name of the key-value table.

    Returns:
    The deserialized data associated with the key.
    """
    retrieve_query = f"SELECT value FROM {table_name} WHERE key = ?;"  # nosec
    try:
        cur = conn.cursor()
        cur.execute(retrieve_query, (key,))
        result = cur.fetchone()
        if result:
            data_retrieved = json.loads(result[0])
            return data_retrieved
        else:
            return None
    except DatabaseError as e:
        log.error(f"Failed to retrieve data for key '{key}': {e}")
        raise
