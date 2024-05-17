from typing import Any, Dict

import pandas as pd
import pytest
from pandas import DataFrame, concat, read_csv
from pandas.api.types import is_float_dtype, is_integer_dtype, is_object_dtype

from gnatss.configs.io import CSVOutput, InputData
from gnatss.configs.main import Configuration
from gnatss.configs.solver import GPSSolutionInput
from gnatss.constants import (
    DEL_ENDTIME,
    DEL_STARTTIME,
    GPS_COV,
    GPS_GEOCENTRIC,
    GPS_TIME,
    L1_DATA_FORMAT,
    RPH_LOCAL_TANGENTS,
    RPH_TIME,
    SP_DEPTH,
    SP_SOUND_SPEED,
    TIME_J2000,
    TT_DATE,
    TT_TIME,
)
from gnatss.loaders import (
    Path,
    load_configuration,
    load_deletions,
    load_gps_solutions,
    load_roll_pitch_heading,
    load_sound_speed,
    load_travel_times,
)
from gnatss.ops.io import gather_files_all_procs
from tests import TEST_DATA_FOLDER


@pytest.fixture
def all_files_dict_j2k_travel_times() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    config.input_files.travel_times = InputData(path="./tests/data/2022/NCL1/**/WG_*/pxp_tt_j2k")
    return gather_files_all_procs(config)


@pytest.fixture
def all_files_dict_legacy_gps_solutions() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    config.solver.input_files.gps_solution = GPSSolutionInput(
        path="./tests/data/2022/NCL1/**/posfilter/POS_FREED_TRANS_TWTT",
        legacy=True,
    )
    return gather_files_all_procs(config)


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


def _load_travel_times_pass_testcase_helper(
    expected_columns, travel_times, transponder_ids, is_j2k, time_scale
):
    loaded_travel_times = load_travel_times(
        files=travel_times,
        transponder_ids=transponder_ids,
        is_j2k=is_j2k,
        time_scale=time_scale,
    )

    # raw_travel_times contains the expected df
    raw_travel_times = concat(
        [read_csv(i, sep=r"\s+", header=None) for i in travel_times]
    ).reset_index(drop=True)

    column_num_diff = len(expected_columns) - len(raw_travel_times.columns)
    if column_num_diff < 0:
        raw_travel_times = raw_travel_times.iloc[:, :column_num_diff]
    raw_travel_times.columns = expected_columns

    if not is_j2k:
        raw_travel_times = raw_travel_times.drop([TT_DATE], axis=1)

    # Assert that df returned from "loaded_travel_times()" matches parameters of expected df
    assert isinstance(loaded_travel_times, DataFrame)
    assert all(
        is_float_dtype(loaded_travel_times[column]) for column in [*transponder_ids, TT_TIME]
    )
    assert loaded_travel_times.shape == raw_travel_times.shape
    assert set(loaded_travel_times.columns.values.tolist()) == set(
        raw_travel_times.columns.values.tolist()
    )

    # Verify microsecond to second conversion for each transponder_id column
    assert loaded_travel_times[transponder_ids].equals(
        raw_travel_times[transponder_ids].apply(lambda x: x * 1e-6)
    )


@pytest.mark.parametrize("is_j2k, time_scale", [(True, "tt"), (False, "tt")])
def test_load_j2k_travel_times(
    transponder_ids, all_files_dict_j2k_travel_times, is_j2k, time_scale
):
    if not is_j2k:
        # load_travel_times() should raise Exception
        # if called with is_j2k=False on j2k type travel time files
        with pytest.raises(AttributeError):
            _ = load_travel_times(
                files=all_files_dict_j2k_travel_times["travel_times"],
                transponder_ids=transponder_ids,
                is_j2k=is_j2k,
                time_scale=time_scale,
            )
    else:
        expected_columns = [TT_TIME, *transponder_ids]
        _load_travel_times_pass_testcase_helper(
            expected_columns,
            all_files_dict_j2k_travel_times["travel_times"],
            transponder_ids,
            is_j2k,
            time_scale,
        )


@pytest.mark.parametrize(
    "is_j2k, time_scale",
    [(True, "tt"), (False, "tt")],
)
def test_load_non_j2k_travel_times(transponder_ids, all_files_dict, is_j2k, time_scale):
    if is_j2k:
        # load_travel_times() should raise Exception
        # if called with is_j2k=True on non-j2k type travel time files
        with pytest.raises(TypeError):
            _ = load_travel_times(
                files=all_files_dict["travel_times"],
                transponder_ids=transponder_ids,
                is_j2k=is_j2k,
                time_scale=time_scale,
            )
    else:
        expected_columns = [TT_DATE, TT_TIME, *transponder_ids]
        _load_travel_times_pass_testcase_helper(
            expected_columns,
            all_files_dict["travel_times"],
            transponder_ids,
            is_j2k,
            time_scale,
        )


@pytest.mark.parametrize(
    "time_round",
    [3, 6],
)
def test_legacy_load_gps_solutions(all_files_dict_legacy_gps_solutions, time_round):
    loaded_gps_solutions = load_gps_solutions(
        all_files_dict_legacy_gps_solutions["gps_solution"],
        time_round,
        from_legacy=True,
    )
    expected_columns = [GPS_TIME, *GPS_GEOCENTRIC, *GPS_COV]

    assert isinstance(loaded_gps_solutions, DataFrame)
    assert set(expected_columns) == set(loaded_gps_solutions.columns.values.tolist())
    assert all(
        is_float_dtype(loaded_gps_solutions[column]) for column in loaded_gps_solutions.columns
    )

    raw_gps_solutions = pd.concat(
        [
            read_csv(i, sep=r"\s+", header=None, names=expected_columns)
            for i in all_files_dict_legacy_gps_solutions["gps_solution"]
        ]
    ).reset_index(drop=True)

    # Dimension of raw_gps_solutions df should equal loaded_gps_solutions df
    assert loaded_gps_solutions.shape == raw_gps_solutions.shape

    # Verify rounding decimal precision of GPS_TIME column
    raw_gps_solutions[GPS_TIME] = raw_gps_solutions[GPS_TIME].round(time_round)
    assert loaded_gps_solutions[GPS_TIME].equals(raw_gps_solutions[GPS_TIME])


def test_load_roll_pitch_heading(all_files_dict_roll_pitch_heading):
    # TODO: Add test for load_roll_pitch_heading data with Covariance rows

    loaded_rph_solutions = load_roll_pitch_heading(
        all_files_dict_roll_pitch_heading["roll_pitch_heading"]
    )
    expected_columns = [RPH_TIME, *RPH_LOCAL_TANGENTS]

    assert isinstance(loaded_rph_solutions, DataFrame)
    assert set(expected_columns) == set(loaded_rph_solutions.columns.values.tolist())
    assert all(
        is_float_dtype(loaded_rph_solutions[column]) for column in loaded_rph_solutions.columns
    )

    raw_rph_solutions = pd.concat(
        [
            read_csv(i, sep=r"\s+", header=None, names=expected_columns)
            for i in all_files_dict_roll_pitch_heading["roll_pitch_heading"]
        ]
    ).reset_index(drop=True)

    # Dimension of raw_rph_solutions df should equal loaded_rph_solutions df
    assert loaded_rph_solutions.shape == raw_rph_solutions.shape


@pytest.fixture()
def create_and_cleanup_outliers_file(
    all_files_dict,
    transponder_ids,
    configuration,
    travel_times_data,
):
    outliers_file = Path(configuration.output.path) / CSVOutput.outliers.value
    deletions_file = Path(configuration.output.path) / CSVOutput.deletions.value
    outliers_df = travel_times_data.sample(frac=0.05)
    outliers_file.unlink(missing_ok=True)
    deletions_file.unlink(missing_ok=True)
    outliers_df.to_csv(outliers_file, index=False)

    yield

    # deletions_file is created by load_deletions()
    deletions_file.unlink()


@pytest.fixture()
def create_and_cleanup_outliers_and_deletions_files(
    all_files_dict,
    transponder_ids,
    configuration,
    travel_times_data,
):
    deletions_file = Path(configuration.output.path) / CSVOutput.deletions.value
    outliers_file = Path(configuration.output.path) / CSVOutput.outliers.value
    outliers_df = travel_times_data.sample(frac=0.05)
    deletions_df = travel_times_data.sample(frac=0.1)
    deletions_df = pd.DataFrame.from_records(
        deletions_df[TT_TIME].apply(lambda row: (row, row)).to_numpy(),
        columns=[DEL_STARTTIME, DEL_ENDTIME],
    )

    outliers_file.unlink(missing_ok=True)
    outliers_df.to_csv(outliers_file, index=False)

    deletions_file.unlink(missing_ok=True)
    deletions_df.to_csv(deletions_file, index=False)

    yield
    # deletions_file is created by load_deletions()
    deletions_file.unlink()


def test_load_deletions_outliers_and_deletions(
    configuration,
    create_and_cleanup_outliers_and_deletions_files,
):
    outliers_file = Path(configuration.output.path) / CSVOutput.outliers.value
    deletions_file = Path(configuration.output.path) / CSVOutput.deletions.value
    outliers_rows = pd.read_csv(outliers_file).shape[0]
    deletions_rows = pd.read_csv(deletions_file).shape[0]

    # Verify outliers_file and deletions_file is present before calling load_deletions()
    assert outliers_file.is_file()
    assert deletions_file.is_file()

    loaded_deletions_df = load_deletions(configuration, None, "tt", remove_outliers=True)

    # Assert concatenation of outliers and deletions df
    assert loaded_deletions_df.shape[0] == outliers_rows + deletions_rows
    assert loaded_deletions_df.columns.tolist() == [DEL_STARTTIME, DEL_ENDTIME]
    assert is_float_dtype(loaded_deletions_df[DEL_STARTTIME]) and is_float_dtype(
        loaded_deletions_df[DEL_ENDTIME]
    )

    # Verify outliers_file is not present and
    # deletions_file is present after calling load_deletions()
    assert not outliers_file.is_file()
    assert deletions_file.is_file()


def test_load_deletions_outliers_only_case(
    configuration,
    create_and_cleanup_outliers_file,
):
    deletions_file = Path(configuration.output.path) / CSVOutput.deletions.value
    outliers_file = Path(configuration.output.path) / CSVOutput.outliers.value
    outliers_rows = pd.read_csv(outliers_file).shape[0]

    # Verify outliers_file is present and
    # deletions_file is not present before calling load_deletions()
    assert outliers_file.is_file()
    assert not deletions_file.is_file()

    loaded_deletions_df = load_deletions(configuration, None, "tt", remove_outliers=True)

    assert loaded_deletions_df.shape[0] == outliers_rows
    assert loaded_deletions_df.columns.tolist() == [DEL_STARTTIME, DEL_ENDTIME]
    assert is_float_dtype(loaded_deletions_df[DEL_STARTTIME]) and is_float_dtype(
        loaded_deletions_df[DEL_ENDTIME]
    )

    # Verify outliers_file is not present and
    # deletions_file is present after calling load_deletions()
    assert not outliers_file.is_file()
    assert deletions_file.is_file()


def test_load_deletions_outliers_and_deletions_from_config(
    configuration,
    create_and_cleanup_outliers_file,
):
    # Use config.yaml to load deletions files
    configuration.solver.input_files.deletions = InputData(path="./tests/data/2022/**/deletns.dat")

    config_deletions_files = gather_files_all_procs(configuration)["deletions"]
    outliers_file = Path(configuration.output.path) / CSVOutput.outliers.value
    deletions_file = Path(configuration.output.path) / CSVOutput.deletions.value

    outliers_rows = pd.read_csv(outliers_file).shape[0]
    config_deletions_rows = sum(
        [
            pd.read_fwf(config_deletions_file, header=None).shape[0]
            for config_deletions_file in config_deletions_files
        ]
    )

    # Verify outliers_file and config_deletions_file is present and
    # output deletions_file is not present before calling load_deletions()
    assert outliers_file.is_file()
    for config_deletions_file in config_deletions_files:
        assert Path(config_deletions_file).is_file()
    assert not deletions_file.is_file()

    loaded_deletions_df = load_deletions(
        configuration, config_deletions_files, "tt", remove_outliers=True
    )

    # Assert concatenation of outliers and deletions df
    assert loaded_deletions_df.shape[0] == outliers_rows + config_deletions_rows
    assert loaded_deletions_df.columns.tolist() == [DEL_STARTTIME, DEL_ENDTIME]
    assert is_float_dtype(loaded_deletions_df[DEL_STARTTIME]) and is_float_dtype(
        loaded_deletions_df[DEL_ENDTIME]
    )

    # Verify outliers_file is not present and
    # deletions_file and config_deletions_file are present after calling load_deletions()
    assert not outliers_file.is_file()
    assert deletions_file.is_file()
    for config_deletions_file in config_deletions_files:
        assert Path(config_deletions_file).is_file()


def test_load_novatel(all_files_dict, novatel_data):
    # Expected columns and its dtypes are defined in L1_DATA_FORMAT.
    # There is an extra generated float column "time".
    expected_columns = list(L1_DATA_FORMAT["INSPVAA"]["data_fields"]) + [TIME_J2000]
    expected_column_dtypes = list(L1_DATA_FORMAT["INSPVAA"]["data_fields_dtypes"]) + ["float"]

    data_files = all_files_dict["novatel"]

    l1_df = novatel_data
    l1_df_columns = l1_df.columns.tolist()
    l1_df_dtypes = l1_df.dtypes.tolist()

    assert l1_df_columns == expected_columns

    for expected_column_dtype, column_dtype in zip(expected_column_dtypes, l1_df_dtypes):
        if expected_column_dtype == "int":
            assert is_integer_dtype(column_dtype)
        elif expected_column_dtype == "float":
            assert is_float_dtype(column_dtype)
        else:
            # Only integer and float dtypes should be present
            assert False

    # TODO: Confirm appropriate missing rows threshold
    missing_rows_threshold_percent = 1
    data_files_rows = 0
    for data_file in data_files:
        with open(data_file, "r") as f:
            data_files_rows += len(f.read().split("\n"))
    minimum_expected_rows = int(((100 - missing_rows_threshold_percent) * data_files_rows) / 100)
    assert l1_df.shape[0] >= minimum_expected_rows


def test_load_novatel_std(
    all_files_dict,
    novatel_std_data,
):
    # Expected columns and its dtypes are defined in L1_DATA_FORMAT.
    # There is an extra generated float column "TIME_J2000".
    expected_columns = list(L1_DATA_FORMAT["INSSTDEVA"]["data_fields"]) + [TIME_J2000]
    expected_column_dtypes = list(L1_DATA_FORMAT["INSSTDEVA"]["data_fields_dtypes"]) + ["float"]

    data_files = all_files_dict["novatel_std"]

    l1_df = novatel_std_data
    l1_df_columns = l1_df.columns.tolist()
    l1_df_dtypes = l1_df.dtypes.tolist()

    assert l1_df_columns == expected_columns

    for expected_column_dtype, column_dtype in zip(expected_column_dtypes, l1_df_dtypes):
        if expected_column_dtype == "int":
            assert is_integer_dtype(column_dtype)
        elif expected_column_dtype == "float":
            assert is_float_dtype(column_dtype)
        elif expected_column_dtype == "object":
            assert is_object_dtype(column_dtype)
        else:
            # Only integer, float, and object dtypes should be present
            assert False

    # TODO: Confirm appropriate missing rows threshold
    missing_rows_threshold_percent = 1
    data_files_rows = 0
    for data_file in data_files:
        with open(data_file, "r") as f:
            data_files_rows += len(f.read().split("\n"))
    minimum_expected_rows = int(((100 - missing_rows_threshold_percent) * data_files_rows) / 100)
    assert l1_df.shape[0] >= minimum_expected_rows


def test_load_novatel_std_v2(
    all_files_dict,
    novatel_std_data,
):
    # Expected columns and its dtypes are defined in L1_DATA_FORMAT.
    # There is an extra generated float column "TIME_J2000".
    expected_columns = list(L1_DATA_FORMAT["INSSTDEVA"]["data_fields"]) + [TIME_J2000]
    expected_column_dtypes = list(L1_DATA_FORMAT["INSSTDEVA"]["data_fields_dtypes"]) + ["float"]

    data_files = all_files_dict["novatel_std"]

    l1_df = novatel_std_data
    l1_df_columns = l1_df.columns.tolist()
    l1_df_dtypes = l1_df.dtypes.tolist()

    assert l1_df_columns == expected_columns

    for expected_column_dtype, column_dtype in zip(expected_column_dtypes, l1_df_dtypes):
        if expected_column_dtype == "int":
            assert is_integer_dtype(column_dtype)
        elif expected_column_dtype == "float":
            assert is_float_dtype(column_dtype)
        elif expected_column_dtype == "object":
            assert is_object_dtype(column_dtype)
        else:
            # Only integer, float, and object dtypes should be present
            assert False

    # TODO: Confirm appropriate missing rows threshold
    missing_rows_threshold_percent = 1
    data_files_rows = 0
    for data_file in data_files:
        with open(data_file, "r") as f:
            data_files_rows += len(f.read().split("\n"))
    minimum_expected_rows = int(((100 - missing_rows_threshold_percent) * data_files_rows) / 100)
    assert l1_df.shape[0] >= minimum_expected_rows
