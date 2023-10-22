from typing import Any, Dict, List

import pytest
from numpy import float64
from pandas import DataFrame, concat, read_csv
from pandas.api.types import is_float_dtype

from gnatss.configs.main import Configuration
from gnatss.constants import SP_DEPTH, SP_SOUND_SPEED, TT_DATE, TT_TIME
from gnatss.loaders import load_configuration, load_sound_speed, load_travel_times
from gnatss.main import gather_files
from tests import TEST_DATA_FOLDER

PARSED_FILE = "parsed"


@pytest.fixture
def all_files_dict() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    return gather_files(config)


@pytest.fixture
def all_files_dict_j2k_travel_times() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    config.solver.input_files.travel_times.path = (
        "./tests/data/2022/NCL1/**/WG_*/pxp_tt_j2k"
    )
    return gather_files(config)


@pytest.mark.parametrize(
    "config_yaml_path",
    [(None), (TEST_DATA_FOLDER / "invalid_config.yaml")],
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
    assert svdf.dtypes[SP_DEPTH] == float64 and svdf.dtypes[SP_SOUND_SPEED] == float64


@pytest.fixture
def transponder_ids() -> List[str]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    transponders = config.solver.transponders
    return [t.pxp_id for t in transponders]


@pytest.mark.parametrize(
    "is_j2k, time_scale",
    [(True, "tt"), (False, "tt")],
)
def test_load_j2k_travel_times(
    transponder_ids, all_files_dict_j2k_travel_times, is_j2k, time_scale
):
    if is_j2k:
        loaded_travel_times = load_travel_times(
            files=all_files_dict_j2k_travel_times["travel_times"],
            transponder_ids=transponder_ids,
            is_j2k=is_j2k,
            time_scale=time_scale,
        )

        # raw_travel_times contains the expected df
        raw_travel_times = concat(
            [
                read_csv(i, delim_whitespace=True, header=None)
                for i in all_files_dict_j2k_travel_times["travel_times"]
                if PARSED_FILE not in i
            ]
        ).reset_index(drop=True)

        expected_columns = [TT_TIME, *transponder_ids]
        column_num_diff = len(expected_columns) - len(raw_travel_times.columns)
        if column_num_diff < 0:
            raw_travel_times = raw_travel_times.iloc[:, :column_num_diff]
        raw_travel_times.columns = expected_columns

        # Assert that df returned from "loaded_travel_times()" matches parameters of expected df
        assert isinstance(loaded_travel_times, DataFrame)
        assert all(
            is_float_dtype(loaded_travel_times[column])
            for column in [*transponder_ids, TT_TIME]
        )
        assert loaded_travel_times.shape == raw_travel_times.shape
        assert set(loaded_travel_times.columns.values.tolist()) == set(
            raw_travel_times.columns.values.tolist()
        )

        # Verify microsecond to second conversion for each transponder_id column
        assert loaded_travel_times[transponder_ids].equals(
            raw_travel_times[transponder_ids].apply(lambda x: x * 1e-6)
        )
    else:
        # load_travel_times() should raise Exception
        # if called with is_j2k=False on j2k type travel time files
        with pytest.raises(AttributeError):
            _ = load_travel_times(
                files=all_files_dict_j2k_travel_times["travel_times"],
                transponder_ids=transponder_ids,
                is_j2k=is_j2k,
                time_scale=time_scale,
            )


@pytest.mark.parametrize(
    "is_j2k, time_scale",
    [(True, "tt"), (False, "tt")],
)
def test_load_non_j2k_travel_times(transponder_ids, all_files_dict, is_j2k, time_scale):
    if not is_j2k:
        expected_columns = [TT_DATE, TT_TIME, *transponder_ids]
        loaded_travel_times = load_travel_times(
            files=all_files_dict["travel_times"],
            transponder_ids=transponder_ids,
            is_j2k=is_j2k,
            time_scale=time_scale,
        )

        # raw_travel_times contains the expected df
        raw_travel_times = concat(
            [
                read_csv(i, delim_whitespace=True, header=None)
                for i in all_files_dict["travel_times"]
                if PARSED_FILE not in i
            ]
        ).reset_index(drop=True)
        column_num_diff = len(expected_columns) - len(raw_travel_times.columns)
        if column_num_diff < 0:
            raw_travel_times = raw_travel_times.iloc[:, :column_num_diff]
        raw_travel_times.columns = expected_columns
        raw_travel_times = raw_travel_times.drop([TT_DATE], axis=1)

        # Assert that df returned from "loaded_travel_times()" matches parameters of expected df
        assert isinstance(loaded_travel_times, DataFrame)
        assert all(
            is_float_dtype(loaded_travel_times[column])
            for column in [*transponder_ids, TT_TIME]
        )
        assert loaded_travel_times.shape == raw_travel_times.shape
        assert set(loaded_travel_times.columns.values.tolist()) == set(
            raw_travel_times.columns.values.tolist()
        )

        # Verify microsecond to second conversion for each transponder_id column
        assert loaded_travel_times[transponder_ids].equals(
            raw_travel_times[transponder_ids].apply(lambda x: x * 1e-6)
        )
    else:
        # load_travel_times() should raise Exception
        # if called with is_j2k=True on non-j2k type travel time files
        with pytest.raises(TypeError):
            _ = load_travel_times(
                files=all_files_dict["travel_times"],
                transponder_ids=transponder_ids,
                is_j2k=is_j2k,
                time_scale=time_scale,
            )
