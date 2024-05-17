from pathlib import Path

from gnatss.configs.main import Configuration


def test_env_configuration_main(blank_env: None) -> None:
    """Testing a simple configuration class
    with environment variables.

    See root/conftest.py for fixture definition.
    """
    config = Configuration()

    assert config.site_id == "test_site"


def test_env_main_posfilter(blank_csv_test_file: Path, blank_env: None) -> None:
    """Testing setting RPS input file path
    with environment variables

    See root/conftest.py for fixture definition.
    """
    test_path = str(blank_csv_test_file)

    config = Configuration()

    # assert config.input_files.travel_times is populated
    assert config.input_files.travel_times.path == test_path
