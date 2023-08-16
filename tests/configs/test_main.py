import os
from pathlib import Path

import pytest

from gnatss.configs.main import Configuration


def test_env_configuration_main():
    os.environ.setdefault("GNATSS_SITE_ID", "test_site")
    with pytest.warns(
        UserWarning,
        match=(
            "Configuration file `config.yaml` not found. "
            "Will attempt to retrieve configuration from environment variables."
        ),
    ):
        config = Configuration()

    assert config.site_id == "test_site"


def test_env_main_posfilter():
    # Create test file
    test_file = Path(__file__).parent / "test_data.csv"
    test_file.touch()

    # Set environment variables
    os.environ.setdefault("GNATSS_SITE_ID", "test_site")
    os.environ.setdefault(
        "GNATSS_POSFILTER__INPUT_FILES__ROLL_PITCH_HEADING__PATH", str(test_file)
    )
    with pytest.warns(
        UserWarning,
        match=(
            "Configuration file `config.yaml` not found. "
            "Will attempt to retrieve configuration from environment variables."
        ),
    ):
        config = Configuration()

    assert config.posfilter is not None
    assert config.posfilter.input_files.roll_pitch_heading.path == str(test_file)

    # Delete test file
    test_file.unlink()
