import os
from pathlib import Path

import pytest


@pytest.fixture
def blank_csv_test_file() -> Path:
    # Create test file
    test_file = Path(__file__).parent / "test_data.csv"
    test_file.touch()
    # This will yield and return until the end of the test
    yield test_file
    # Delete test file
    test_file.unlink()


@pytest.fixture
def blank_env(blank_csv_test_file: Path) -> None:
    # Set environment variables
    os.environ.setdefault("GNATSS_SITE_ID", "test_site")
    os.environ.setdefault(
        "GNATSS_POSFILTER__INPUT_FILES__ROLL_PITCH_HEADING__PATH",
        str(blank_csv_test_file),
    )
