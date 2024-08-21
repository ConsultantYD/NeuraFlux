import sqlite3

import pandas as pd
import pytest

from neuraflux.utils_sql import (
    add_dataframe_to_table,
    check_table_exists,
    create_connection_to_db,
    infer_sqlite_type,
    create_key_value_table,
    store_value_in_db,
    retrieve_value_from_db,
)


@pytest.fixture
def db_connection():
    """Fixture to provide a SQLite database connection and clean up after the test."""
    # Create an in-memory SQLite database
    connection = sqlite3.connect(":memory:")
    yield connection
    # Close the database connection after each test
    connection.close()


def test_check_table_exists(db_connection):
    # Create a table directly via SQL for control
    cursor = db_connection.cursor()
    cursor.execute("CREATE TABLE TestTable (id INTEGER PRIMARY KEY, value TEXT)")
    db_connection.commit()

    # Check for the table existence
    assert check_table_exists(db_connection, "TestTable")
    assert not check_table_exists(db_connection, "NonexistentTable")


@pytest.fixture
def file_db_connection(tmp_path):
    """Fixture for creating and deleting a file-based SQLite database."""
    db_file = tmp_path / "test.db"
    connection = sqlite3.connect(str(db_file))
    yield connection
    connection.close()
    db_file.unlink(missing_ok=True)


def test_create_connection_to_db(file_db_connection):
    # Test file-based connection cleanup
    assert isinstance(file_db_connection, sqlite3.Connection)

    # Test in-memory connection
    conn = create_connection_to_db(":memory:")
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_create_connection_to_db_exception():
    # Test with an invalid database file path (simulating an error)
    with pytest.raises(sqlite3.DatabaseError):
        create_connection_to_db("/invalid_path/invalid.db")


def test_infer_sqlite_type():
    assert infer_sqlite_type(pd.Series([1, 2, 3]).dtype) == "INTEGER"
    assert infer_sqlite_type(pd.Series([1.0, 2.0, 3.0]).dtype) == "REAL"
    assert (
        infer_sqlite_type(pd.to_datetime(["2021-01-01", "2021-01-02"]).dtype)
        == "DATETIME"
    )
    assert infer_sqlite_type(pd.Series([True, False]).dtype) == "BOOLEAN"
    assert infer_sqlite_type(pd.Series(["text", "more text"]).dtype) == "TEXT"
    # Tests for edge cases or unusual data types
    assert (
        infer_sqlite_type(pd.Series([None, "mixed", 1, 2.5]).dtype) == "TEXT"
    )  # Mixed types default to TEXT


def test_add_dataframe_to_table(db_connection):
    # Prepare the DataFrame
    data = {"timestamp": [1, 2], "Name": ["Alice", "Bob"], "Age": [25, 30]}
    df = pd.DataFrame(data)

    # Insert the DataFrame into the newly created table
    add_dataframe_to_table(df, db_connection, "People", "timestamp")

    # Verify that the data was inserted correctly
    cursor = db_connection.cursor()
    cursor.execute("SELECT timestamp, Name, Age FROM People")
    rows = cursor.fetchall()

    # Convert fetched data back to a DataFrame for easy comparison
    fetched_df = pd.DataFrame(rows, columns=["timestamp", "Name", "Age"])

    # Check if the DataFrame created from the fetched data matches the original DataFrame
    pd.testing.assert_frame_equal(
        df.sort_values(by="Name").reset_index(drop=True),
        fetched_df.sort_values(by="Name").reset_index(drop=True),
    )

    # Ensure that all the data from the DataFrame exists in the table
    assert len(rows) == len(df)


def test_create_key_value_table(db_connection):
    create_key_value_table(db_connection, "key_value_store")

    # Verify the table was created
    assert check_table_exists(db_connection, "key_value_store")

    # Check if the schema is correct
    cursor = db_connection.cursor()
    cursor.execute("PRAGMA table_info(key_value_store)")
    columns = cursor.fetchall()
    assert len(columns) == 2  # Check for two columns (key, value)
    assert (
        columns[0][1] == "key" and columns[0][2] == "TEXT"
    )  # Key column exists and is TEXT type
    assert (
        columns[1][1] == "value" and columns[1][2] == "BLOB"
    )  # Value column exists and is BLOB type


def test_store_and_retrieve_value(db_connection):
    create_key_value_table(db_connection, "key_value_store")

    # Data to be stored
    test_key = "user1"
    test_value = {"name": "Alice", "age": 30, "active": True}

    # Store the value
    store_value_in_db(db_connection, test_key, test_value, "key_value_store")

    # Retrieve the value
    retrieved_value = retrieve_value_from_db(db_connection, test_key, "key_value_store")

    # Verify the retrieved data matches the original data
    assert (
        retrieved_value == test_value
    ), "The retrieved value should match the original data"

    # Test retrieval of a non-existent key
    non_existent_value = retrieve_value_from_db(
        db_connection, "non_existent_key", "key_value_store"
    )
    assert non_existent_value is None, "Expected None for a non-existent key"


@pytest.mark.parametrize(
    "key,value",
    [
        ("test_int", 123),
        ("test_str", "hello"),
        ("test_dict", {"key": "value", "number": 42}),
        ("test_list", [1, 2, 3, "four"]),
    ],
)
def test_store_and_retrieve_various_types(db_connection, key, value):
    create_key_value_table(db_connection, "key_value_store")

    # Store different types of data
    store_value_in_db(db_connection, key, value, "key_value_store")

    # Retrieve and assert correctness
    assert (
        retrieve_value_from_db(db_connection, key, "key_value_store") == value
    ), f"Data mismatch for key {key} with value {value}"


def test_insert_initial_dataframe(db_connection):
    # Prepare initial DataFrame
    data = {
        "index_col": [1, 2],
        "A": [10, 20],
        "B": [30, 40],
        "C": [50, 60],
        "D": [None, None],
        "E": [None, 100],
    }
    df = pd.DataFrame(data)

    table_name = "test_table"

    # Create table and insert DataFrame
    add_dataframe_to_table(df, db_connection, table_name, index_col="index_col")

    # Verify that data was inserted correctly
    cursor = db_connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    assert len(rows) == 2, "Expected two rows in the database after initial insert"
