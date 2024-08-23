import datetime as dt
import os
import json
import tempfile

import pytest
from neuraflux.global_variables import DT_STR_FORMAT
from neuraflux.time_ref import TimeRef


@pytest.fixture
def sample_datetime():
    return dt.datetime(2024, 4, 22, 12, 0, 0)


@pytest.fixture
def time_ref(sample_datetime):
    return TimeRef(start_time_utc=sample_datetime)


def test_initialization(time_ref, sample_datetime):
    assert time_ref.get_time_utc() == sample_datetime
    assert time_ref.get_initial_time_utc() == sample_datetime


def test_get_time_utc_as_str(time_ref, sample_datetime):
    expected_time_str = sample_datetime.strftime(DT_STR_FORMAT)
    assert time_ref.get_time_utc_as_str() == expected_time_str


def test_get_initial_time_utc_as_str(time_ref, sample_datetime):
    expected_time_str = sample_datetime.strftime(DT_STR_FORMAT)
    assert time_ref.get_initial_time_utc_as_str() == expected_time_str


def test_increment_time_by_timedelta(time_ref, sample_datetime):
    delta = dt.timedelta(hours=1)
    time_ref.increment_time(delta)
    assert time_ref.get_time_utc() == sample_datetime + delta


def test_increment_time_by_int(time_ref, sample_datetime):
    seconds = 3600
    time_ref.increment_time(seconds)
    assert time_ref.get_time_utc() == sample_datetime + dt.timedelta(seconds=seconds)


def test_to_file(time_ref):
    with tempfile.TemporaryDirectory() as tmpdirname:
        time_ref.to_file(tmpdirname)
        expected_file_path = os.path.join(tmpdirname, "time_ref.json")
        with open(expected_file_path, "r") as file:
            data = json.load(file)
        assert data["time_utc"] == time_ref.get_time_utc_as_str()
        assert data["initial_time_utc"] == time_ref.get_initial_time_utc_as_str()
        # No need to delete file explicitly; tempfile handles this


def test_from_file(sample_datetime):
    with tempfile.TemporaryDirectory() as tmpdirname:
        initial_time_ref = TimeRef(start_time_utc=sample_datetime)
        initial_time_ref.to_file(tmpdirname)
        loaded_time_ref = TimeRef.from_file(tmpdirname)
        assert loaded_time_ref.get_time_utc() == sample_datetime
        assert loaded_time_ref.get_initial_time_utc() == sample_datetime
        # No need to delete file explicitly; tempfile handles this
