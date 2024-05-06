import os
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import pytest
from pandas import DataFrame

from gnatss.configs.io import InputData
from gnatss.configs.main import Configuration
from gnatss.loaders import (
    Path,
    load_configuration,
    load_novatel,
    load_novatel_std,
    load_roll_pitch_heading,
    load_travel_times,
)
from gnatss.ops.io import gather_files_all_procs
from tests import TEST_DATA_FOLDER


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


@pytest.fixture(scope="session")
def configuration() -> Configuration:
    return load_configuration(TEST_DATA_FOLDER / "config.yaml")


@pytest.fixture(scope="session")
def all_files_dict() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    return gather_files_all_procs(config)


@pytest.fixture(scope="session")
def novatel_data(all_files_dict) -> DataFrame:
    data_files = all_files_dict["novatel"]
    return load_novatel(data_files)


@pytest.fixture(scope="session")
def novatel_std_data(all_files_dict) -> DataFrame:
    data_files = all_files_dict["novatel_std"]
    return load_novatel_std(data_files)


@pytest.fixture(scope="session")
def transponder_ids(configuration) -> List[str]:
    transponders = configuration.solver.transponders
    return [t.pxp_id for t in transponders]


@pytest.fixture(scope="session")
def travel_times_data(all_files_dict, transponder_ids) -> DataFrame:
    return load_travel_times(
        files=all_files_dict["travel_times"],
        transponder_ids=transponder_ids,
        is_j2k=False,
        time_scale="tt",
    )


@pytest.fixture(scope="session")
def all_files_dict_roll_pitch_heading() -> Dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    config.posfilter.input_files.roll_pitch_heading = InputData(
        path="./tests/data/2022/NCL1/**/WG_*/RPH_TWTT",
    )
    return gather_files_all_procs(config)


@pytest.fixture(scope="session")
def roll_pitch_heading_data(all_files_dict_roll_pitch_heading) -> DataFrame:
    return load_roll_pitch_heading(
        all_files_dict_roll_pitch_heading["roll_pitch_heading"]
    )
