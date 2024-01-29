import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest


@pytest.fixture
def blank_csv_test_file() -> Path:
    with NamedTemporaryFile(suffix=".csv") as f:
        f.write(b"")
        f.flush()
        yield Path(f.name)


@pytest.fixture
def blank_env(blank_csv_test_file: Path) -> None:
    blank_envs = {
        "GNATSS_SITE_ID": "test_site",
        "GNATSS_POSFILTER__INPUT_FILES__ROLL_PITCH_HEADING__PATH": str(
            blank_csv_test_file
        ),
        "GNATSS_POSFILTER__ATD_OFFSETS__FORWARD": str(0.000053),
        "GNATSS_POSFILTER__ATD_OFFSETS__RIGHTWARD": str(0),
        "GNATSS_POSFILTER__ATD_OFFSETS__DOWNWARD": str(0.92813),
    }
    for k, v in blank_envs.items():
        os.environ.setdefault(k, v)

    yield

    # Clean up environment variables
    for k, v in blank_envs.items():
        os.environ.pop(k)
