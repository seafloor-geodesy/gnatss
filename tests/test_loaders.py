from typing import Any, Dict

import pytest
from numpy import float64
from pandas import DataFrame

from gnatss.configs.main import Configuration
from gnatss.constants import SP_DEPTH, SP_SOUND_SPEED
from gnatss.loaders import load_configuration, load_sound_speed
from gnatss.main import gather_files
from tests import TEST_DATA_FOLDER


@pytest.fixture
def all_files_dict() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
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
