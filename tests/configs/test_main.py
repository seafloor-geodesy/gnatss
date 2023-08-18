from pathlib import Path

import pytest

from gnatss.configs.main import Configuration


def test_env_configuration_main(blank_env: None) -> None:
    """Testing a simple configuration class
    with environment variables.

    See root/conftest.py for fixture definition.
    """
    with pytest.warns(
        UserWarning,
        match=(
            "Configuration file `config.yaml` not found. "
            "Will attempt to retrieve configuration from environment variables."
        ),
    ):
        config = Configuration()

    assert config.site_id == "test_site"


def test_env_main_posfilter(blank_csv_test_file: Path, blank_env: None) -> None:
    """Testing setting RPS input file path
    with environment variables

    See root/conftest.py for fixture definition.
    """
    test_path = str(blank_csv_test_file)

    with pytest.warns(
        UserWarning,
        match=(
            "Configuration file `config.yaml` not found. "
            "Will attempt to retrieve configuration from environment variables."
        ),
    ):
        config = Configuration()

    assert config.posfilter is not None
    assert config.posfilter.input_files.roll_pitch_heading.path == test_path
