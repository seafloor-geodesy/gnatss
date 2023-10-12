from typing import Any, Dict

import pytest
from numpy import float64
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


# @pytest.mark.parametrize(
#     "config_yaml_path",
#     [(None), (TEST_DATA_FOLDER / "invalid_config.yaml")],
# )
# def test_load_configuration_invalid_path(config_yaml_path):
#     if config_yaml_path is None:
#         with pytest.raises(FileNotFoundError):
#             load_configuration(config_yaml_path)
#
#
# @pytest.mark.parametrize(
#     "config_yaml_path",
#     [TEST_DATA_FOLDER / "config.yaml"],
# )
# def test_load_configuration_valid_path(config_yaml_path):
#     config = load_configuration(config_yaml_path)
#     assert isinstance(config, Configuration)
#
#
# def test_load_sound_speed(all_files_dict):
#     svdf = load_sound_speed(all_files_dict["sound_speed"])
#     assert isinstance(svdf, DataFrame)
#     assert {SP_DEPTH, SP_SOUND_SPEED} == set(svdf.columns.values.tolist())
#     assert is_float_dtype(svdf[SP_DEPTH]) and is_float_dtype(svdf[SP_SOUND_SPEED])


@pytest.mark.parametrize(
    "time_round",
    [
        (3),
    ],
)
def test_load_gps_solutions(all_files_dict, time_round):
    all_gps_solutions = load_gps_solutions(all_files_dict["gps_solution"], time_round)
    expected_columns = [GPS_TIME, *GPS_GEOCENTRIC, *GPS_COV]

    assert isinstance(all_gps_solutions, DataFrame)
    assert set(expected_columns) == set(all_gps_solutions.columns.values.tolist())
    assert all(
        is_float_dtype(all_gps_solutions[column])
        for column in all_gps_solutions.columns
    )

    # Rows in all_gps_solutions df should equal sum of rows in each individual_gps_solutions df
    individual_gps_solutions = [
        read_csv(i, delim_whitespace=True, header=None, names=expected_columns)
        for i in all_files_dict["gps_solution"]
    ]
    assert all_gps_solutions.shape[0] == sum(
        [df.shape[0] for df in individual_gps_solutions]
    )

    # verify that GPS_TIME column has been rounded to time_round accuracy
    import warnings

    import pandas as pd
    from pytest import approx

    new_df = all_gps_solutions[
        [GPS_TIME, GPS_TIME]
    ].copy()  # pd.DataFrame([all_gps_solutions[GPS_TIME], all_gps_solutions[GPS_TIME].round(time_round)])
    new_df.columns = [GPS_TIME, f"Rounded_{GPS_TIME}"]
    new_df.loc[:, f"Rounded_{GPS_TIME}"] = new_df[GPS_TIME].round(time_round)
    with pd.option_context("display.float_format", "{:0.20f}".format):
        print(new_df)
    # warnings.warn(f"CONDITION: {(all_gps_solutions[GPS_TIME] == approx(all_gps_solutions[GPS_TIME].round(time_round))).all()} ")
    assert (new_df[GPS_TIME] == approx(new_df[f"Rounded_{GPS_TIME}"])).all()
