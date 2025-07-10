from __future__ import annotations

import os
from json import dumps
from tempfile import NamedTemporaryFile
from typing import Any

import pytest
from pandas import DataFrame

from gnatss.configs.io import InputData
from gnatss.configs.main import Configuration
from gnatss.configs.solver import GPSSolutionInput
from gnatss.loaders import (
    Path,
    load_configuration,
    load_gps_positions,
    load_gps_solutions,
    load_novatel,
    load_novatel_std,
    load_roll_pitch_heading,
    load_travel_times,
)
from gnatss.ops.io import gather_files_all_procs
from gnatss.posfilter import kalman_filtering, spline_interpolate
from tests import TEST_DATA_FOLDER


@pytest.fixture()
def blank_csv_test_file() -> Path:
    with NamedTemporaryFile(suffix=".csv") as f:
        f.write(b"")
        f.flush()
        yield Path(f.name)


@pytest.fixture()
def blank_env(blank_csv_test_file: Path) -> None:
    blank_envs = {
        "GNATSS_SITE_ID": "test_site",
        "GNATSS_ARRAY_CENTER__LAT": str(1.1),
        "GNATSS_ARRAY_CENTER__LON": str(2.2),
        "GNATSS_TRANSPONDERS": "[]",
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
def all_files_dict() -> dict[str, Any]:
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
def gps_positions_data(all_files_dict) -> DataFrame:
    data_files = all_files_dict["gps_positions"]
    return load_gps_positions(data_files)


@pytest.fixture(scope="session")
def transponder_ids(configuration) -> list[str]:
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
def all_files_dict_roll_pitch_heading() -> dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    config.posfilter.input_files.roll_pitch_heading = InputData(
        path="./tests/data/2022/NCL1/**/WG_*/RPH_TWTT",
    )
    return gather_files_all_procs(config)


@pytest.fixture(scope="session")
def roll_pitch_heading_data(all_files_dict_roll_pitch_heading) -> DataFrame:
    return load_roll_pitch_heading(all_files_dict_roll_pitch_heading["roll_pitch_heading"])


@pytest.fixture(scope="session")
def spline_interpolate_data(
    novatel_data,
    novatel_std_data,
    travel_times_data,
):
    return spline_interpolate(novatel_data, novatel_std_data, travel_times_data, full_result=False)


@pytest.fixture(scope="session")
def _data(
    novatel_data,
    novatel_std_data,
    travel_times_data,
):
    return spline_interpolate(novatel_data, novatel_std_data, travel_times_data, full_result=False)


@pytest.fixture(scope="session")
def kalman_filtering_data(
    novatel_data,
    novatel_std_data,
    gps_positions_data,
    travel_times_data,
):
    kalman_data = kalman_filtering(
        inspvaa_df=novatel_data,
        insstdeva_df=novatel_std_data,
        gps_df=gps_positions_data,
        twtt_df=travel_times_data,
    )
    return kalman_data


@pytest.fixture(scope="session")
def all_files_dict_legacy_gps_solutions() -> dict[str, Any]:
    config = load_configuration(TEST_DATA_FOLDER / "config.yaml")
    config.solver.input_files.gps_solution = GPSSolutionInput(
        path="./tests/data/2022/NCL1/**/posfilter/POS_FREED_TRANS_TWTT",
        legacy=True,
    )
    return gather_files_all_procs(config)


@pytest.fixture(scope="session")
def legacy_gps_solutions_data(all_files_dict_legacy_gps_solutions):
    return load_gps_solutions(
        all_files_dict_legacy_gps_solutions["gps_solution"],
        from_legacy=True,
    )
