from typing import Any, Dict, List

import pandas as pd
import pytest
from pandas import DataFrame, concat, read_csv
from pandas.api.types import is_float_dtype

from gnatss.configs.main import Configuration
from gnatss.constants import (
    GPS_COV,
    GPS_GEOCENTRIC,
    GPS_TIME,
    SP_DEPTH,
    SP_SOUND_SPEED,
    TT_DATE,
    TT_TIME,
)
from gnatss.loaders import (
    load_configuration,
    load_deletions,
    load_gps_solutions,
    load_sound_speed,
    load_travel_times,
)
from gnatss.main import gather_files
from tests import TEST_DATA_FOLDER

PARSED_FILE = "parsed"


@pytest.fixture
def configuration() -> Dict[str, Any]:
    return load_configuration(TEST_DATA_FOLDER / "config.yaml")


@pytest.fixture
def all_files_dict() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    return gather_files(config)


#
# @pytest.fixture
# def all_files_dict_j2k_travel_times() -> Dict[str, Any]:
#     config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
#     config.solver.input_files.travel_times.path = (
#         "./tests/data/2022/NCL1/**/WG_*/pxp_tt_j2k"
#     )
#     return gather_files(config)
#
#
# @pytest.mark.parametrize(
#     "config_yaml_path",
#     [None, TEST_DATA_FOLDER / "invalid_config.yaml"],
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


@pytest.fixture
def transponder_ids() -> List[str]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    transponders = config.solver.transponders
    return [t.pxp_id for t in transponders]


# def _load_travel_times_pass_testcase_helper(
#     expected_columns, travel_times, transponder_ids, is_j2k, time_scale
# ):
#     loaded_travel_times = load_travel_times(
#         files=travel_times,
#         transponder_ids=transponder_ids,
#         is_j2k=is_j2k,
#         time_scale=time_scale,
#     )
#
#     # raw_travel_times contains the expected df
#     raw_travel_times = concat(
#         [
#             read_csv(i, delim_whitespace=True, header=None)
#             for i in travel_times
#             if PARSED_FILE not in i
#         ]
#     ).reset_index(drop=True)
#
#     column_num_diff = len(expected_columns) - len(raw_travel_times.columns)
#     if column_num_diff < 0:
#         raw_travel_times = raw_travel_times.iloc[:, :column_num_diff]
#     raw_travel_times.columns = expected_columns
#
#     if not is_j2k:
#         raw_travel_times = raw_travel_times.drop([TT_DATE], axis=1)
#
#     # Assert that df returned from "loaded_travel_times()" matches parameters of expected df
#     assert isinstance(loaded_travel_times, DataFrame)
#     assert all(
#         is_float_dtype(loaded_travel_times[column])
#         for column in [*transponder_ids, TT_TIME]
#     )
#     assert loaded_travel_times.shape == raw_travel_times.shape
#     assert set(loaded_travel_times.columns.values.tolist()) == set(
#         raw_travel_times.columns.values.tolist()
#     )
#
#     # Verify microsecond to second conversion for each transponder_id column
#     assert loaded_travel_times[transponder_ids].equals(
#         raw_travel_times[transponder_ids].apply(lambda x: x * 1e-6)
#     )
#
#
# @pytest.mark.parametrize(
#     "is_j2k, time_scale",
#     [(True, "tt"), (False, "tt")],
# )
# def test_load_j2k_travel_times(
#     transponder_ids, all_files_dict_j2k_travel_times, is_j2k, time_scale
# ):
#     if not is_j2k:
#         # load_travel_times() should raise Exception
#         # if called with is_j2k=False on j2k type travel time files
#         with pytest.raises(AttributeError):
#             _ = load_travel_times(
#                 files=all_files_dict_j2k_travel_times["travel_times"],
#                 transponder_ids=transponder_ids,
#                 is_j2k=is_j2k,
#                 time_scale=time_scale,
#             )
#     else:
#         expected_columns = [TT_TIME, *transponder_ids]
#         _load_travel_times_pass_testcase_helper(
#             expected_columns,
#             all_files_dict_j2k_travel_times["travel_times"],
#             transponder_ids,
#             is_j2k,
#             time_scale,
#         )
#
#
# @pytest.mark.parametrize(
#     "is_j2k, time_scale",
#     [(True, "tt"), (False, "tt")],
# )
# def test_load_non_j2k_travel_times(transponder_ids, all_files_dict, is_j2k, time_scale):
#     if is_j2k:
#         # load_travel_times() should raise Exception
#         # if called with is_j2k=True on non-j2k type travel time files
#         with pytest.raises(TypeError):
#             _ = load_travel_times(
#                 files=all_files_dict["travel_times"],
#                 transponder_ids=transponder_ids,
#                 is_j2k=is_j2k,
#                 time_scale=time_scale,
#             )
#     else:
#         expected_columns = [TT_DATE, TT_TIME, *transponder_ids]
#         _load_travel_times_pass_testcase_helper(
#             expected_columns,
#             all_files_dict["travel_times"],
#             transponder_ids,
#             is_j2k,
#             time_scale,
#         )
#
#
# @pytest.mark.parametrize(
#     "time_round",
#     [3, 6],
# )
# def test_load_gps_solutions(all_files_dict, time_round):
#     loaded_gps_solutions = load_gps_solutions(
#         all_files_dict["gps_solution"], time_round
#     )
#     expected_columns = [GPS_TIME, *GPS_GEOCENTRIC, *GPS_COV]
#
#     assert isinstance(loaded_gps_solutions, DataFrame)
#     assert set(expected_columns) == set(loaded_gps_solutions.columns.values.tolist())
#     assert all(
#         is_float_dtype(loaded_gps_solutions[column])
#         for column in loaded_gps_solutions.columns
#     )
#
#     raw_gps_solutions = pd.concat(
#         [
#             read_csv(i, delim_whitespace=True, header=None, names=expected_columns)
#             for i in all_files_dict["gps_solution"]
#         ]
#     ).reset_index(drop=True)
#
#     # Dimension of raw_gps_solutions df should equal loaded_gps_solutions df
#     assert loaded_gps_solutions.shape == raw_gps_solutions.shape
#
#     # Verify rounding decimal precision of GPS_TIME column
#     raw_gps_solutions[GPS_TIME] = raw_gps_solutions[GPS_TIME].round(time_round)
#     assert loaded_gps_solutions[GPS_TIME].equals(raw_gps_solutions[GPS_TIME])
#
from gnatss.loaders import Path


@pytest.mark.parametrize(
    "deletions_op_file_present, outliers_op_file_present",
    [
        # (False, False),
        # (True, False),
        (False, True),
        # (False, False),
    ],
)
def test_load_deletions(
    mocker,
    all_files_dict,
    transponder_ids,
    configuration,
    deletions_op_file_present,
    outliers_op_file_present,
):
    from gnatss.configs.io import CSVOutput

    def side_effect(*args, **kwargs):
        return {
            Path(configuration.output.path)
            / CSVOutput.deletions.value: deletions_op_file_present,
            Path(configuration.output.path)
            / CSVOutput.outliers.value: outliers_op_file_present,
        }[args[0]]

    mocker.patch.object(
        Path,
        "exists",
        side_effect=side_effect,
    )
    loaded_travel_times = load_travel_times(
        files=all_files_dict["travel_times"],
        transponder_ids=transponder_ids,
        is_j2k=False,
        time_scale="tt",
    )
    sampled_travel_times = loaded_travel_times.sample(frac=0.1)

    loaded_cut_df = load_deletions(None, configuration, "tt")
    sampled_cut_df = loaded_cut_df.sample(frac=0.1)

    # raw_cut_df = pd.concat(
    #     [
    #         read_csv(i, delim_whitespace=True, header=None, names=expected_columns)
    #         for i in all_files_dict["gps_solution"]
    #     ]
    # ).reset_index(drop=True)
    #
    # # Dimension of raw_gps_solutions df should equal loaded_gps_solutions df
    # assert loaded_gps_solutions.shape == raw_gps_solutions.shape
    #
    # # Verify rounding decimal precision of GPS_TIME column
    # raw_gps_solutions[GPS_TIME] = raw_gps_solutions[GPS_TIME].round(time_round)
    # assert loaded_gps_solutions[GPS_TIME].equals(raw_gps_solutions[GPS_TIME])
