import pytest

from gnatss.configs.main import Configuration
from gnatss.loaders import load_configuration
from tests import TEST_DATA_FOLDER


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
