import os
from json import dumps
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
        "GNATSS_ARRAY_CENTER__LAT": str(1.1),
        "GNATSS_ARRAY_CENTER__LON": str(2.2),
        "GNATSS_TRANSPONDERS": "[]",
        "GNATSS_INPUT_FILES": dumps(
            {"travel_times": {"path": str(blank_csv_test_file)}}
        ),
    }
    for k, v in blank_envs.items():
        os.environ.setdefault(k, v)

    yield

    # Clean up environment variables
    for k, v in blank_envs.items():
        os.environ.pop(k)
