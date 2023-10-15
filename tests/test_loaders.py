from typing import Any, Dict

import pandas as pd
import pytest
from numpy import isclose
from pandas import DataFrame, read_csv
from pandas.api.types import is_float_dtype

from gnatss.configs.main import Configuration
from gnatss.constants import GPS_COV, GPS_GEOCENTRIC, GPS_TIME, SP_DEPTH, SP_SOUND_SPEED
from gnatss.loaders import load_configuration, load_gps_solutions, load_sound_speed
from gnatss.main import gather_files
from tests import TEST_DATA_FOLDER


@pytest.fixture
def all_files_dict() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    return gather_files(config)


@pytest.mark.parametrize(
    "config_yaml_path",
    [None, TEST_DATA_FOLDER / "invalid_config.yaml"],
)
def test_load_configuration_invalid_path(config_yaml_path):
    if config_yaml_path is None:
        with pytest.raises(FileNotFoundError):
            load_configuration(config_yaml_path)


@pytest.mark.parametrize(
    "config_yaml_path",
    [TEST_DATA_FOLDER / "config.yaml"],
)
def test_load_configuration_valid_path(config_yaml_path):
    config = load_configuration(config_yaml_path)
    assert isinstance(config, Configuration)


def test_load_sound_speed(all_files_dict):
    svdf = load_sound_speed(all_files_dict["sound_speed"])
    assert isinstance(svdf, DataFrame)
    assert {SP_DEPTH, SP_SOUND_SPEED} == set(svdf.columns.values.tolist())
    assert is_float_dtype(svdf[SP_DEPTH]) and is_float_dtype(svdf[SP_SOUND_SPEED])


@pytest.mark.parametrize(
    "time_round",
    [3, 6],
)
def test_load_gps_solutions(all_files_dict, time_round):
    loaded_gps_solutions = load_gps_solutions(
        all_files_dict["gps_solution"], time_round
    )
    expected_columns = [GPS_TIME, *GPS_GEOCENTRIC, *GPS_COV]

    assert isinstance(loaded_gps_solutions, DataFrame)
    assert set(expected_columns) == set(loaded_gps_solutions.columns.values.tolist())
    assert all(
        is_float_dtype(loaded_gps_solutions[column])
        for column in loaded_gps_solutions.columns
    )

    raw_gps_solutions = pd.concat(
        [
            read_csv(i, delim_whitespace=True, header=None, names=expected_columns)
            for i in all_files_dict["gps_solution"]
        ]
    ).reset_index(drop=True)

    # Dimension of raw_gps_solutions df should equal loaded_gps_solutions df
    assert loaded_gps_solutions.shape == raw_gps_solutions.shape

    # Verify rounding decimal precision of GPS_TIME column
    gps_times = pd.concat(
        [loaded_gps_solutions[GPS_TIME], raw_gps_solutions[GPS_TIME]],
        axis=1,
        keys=[f"loaded_gps_solutions_{GPS_TIME}", f"raw_gps_solutions_{GPS_TIME}"],
    )
    gps_times["equality"] = gps_times.apply(
        lambda row: isclose(
            row[f"loaded_gps_solutions_{GPS_TIME}"],
            row[f"raw_gps_solutions_{GPS_TIME}"].round(time_round),
        ),
        axis=1,
    )
    assert gps_times["equality"].all()
