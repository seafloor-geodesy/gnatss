from pathlib import Path
from unittest.mock import call

import pytest

from gnatss.configs.io import InputData
from gnatss.configs.solver import SolverTransponder
from gnatss.ops.io import gather_files
from gnatss.solver.utilities import _get_latest_process, _print_final_stats


@pytest.mark.parametrize(
    "proc, mode",
    [
        ("posfilter", "files"),
        ("posfilter", "object"),
        ("unrecognized_proc", "files"),
        ("unrecognized_proc", "object"),
    ],
)
def test_gather_files(configuration, proc, mode):
    if proc == "unrecognized_proc":
        with pytest.raises(AttributeError) as exc_info:
            _ = gather_files(configuration, proc=proc, mode=mode)
        assert str(exc_info.value) == f"Unknown process type: {proc}"

    elif proc == "posfilter":
        all_files_dict = gather_files(configuration, proc=proc, mode=mode)
        print(f"{all_files_dict=}")

        if mode == "files":
            assert isinstance(all_files_dict, dict)
            assert all(
                key in all_files_dict.keys() for key in ("gps_positions", "novatel", "novatel_std")
            )
            assert (
                len(all_files_dict["gps_positions"]) == 3
                and len(all_files_dict["novatel"]) == 1
                and len(all_files_dict["novatel_std"]) == 1
            )

            for key, val in all_files_dict.items():
                assert isinstance(val, list)
                assert isinstance(key, str)
                assert all(isinstance(file, str) for file in val)
                assert all(Path(file).is_file() for file in val)

        elif mode == "object":
            assert isinstance(all_files_dict, dict)
            assert all(
                key in all_files_dict.keys() for key in ("gps_positions", "novatel", "novatel_std")
            )
            for key, val in all_files_dict.items():
                assert isinstance(val, InputData)
                assert isinstance(key, str)

        else:
            assert False

    else:
        assert False


def test__get_latest_process():
    """Test helper function to get the latest process dictionary"""
    # The actual data input doesn't matter,
    # just need to test it's getting the dictionary
    # with the highest key
    process_data = {
        1: {"status": "success", "result": 10},
        2: {"status": "success", "result": 20},
        3: {"status": "failed", "result": None},
        4: {"status": "success", "result": 30},
    }
    latest_process = _get_latest_process(process_data)
    assert latest_process == {"status": "success", "result": 30}


def test__print_final_stats(mocker):
    """Test helper function to print final stats"""
    # Patch typer echo
    mock_echo = mocker.patch("typer.echo")
    # Create some mock data
    transponders = [
        SolverTransponder(pxp_id="1", lat=1.1, lon=2.1, height=3.1, internal_delay=0.01),
        SolverTransponder(pxp_id="2", lat=1.2, lon=2.2, height=3.2, internal_delay=0.02),
        SolverTransponder(pxp_id="3", lat=1.3, lon=2.3, height=3.3, internal_delay=0.03),
    ]
    process_data = {
        1: {
            "data": {"sigpx": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]},
            "transponders_lla": [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)],
            "enu": [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0), (70.0, 80.0, 90.0)],
            "sig_enu": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)],
            "transponders_xyz": [
                (100.0, 200.0, 300.0),
                (400.0, 500.0, 600.0),
                (700.0, 800.0, 900.0),
            ],
        },
        2: {
            "data": {"sigpx": [1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 8.0, 9.0]},
            "transponders_lla": [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)],
            "enu": [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0), (70.0, 80.0, 90.0)],
            "sig_enu": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)],
            "transponders_xyz": [
                (100.0, 200.0, 300.0),
                (400.0, 500.0, 600.0),
                (700.0, 800.0, 900.0),
            ],
        },
    }

    # Call the function
    _print_final_stats(transponders, process_data)

    # Check the output
    expected_calls = [
        call("---- FINAL SOLUTION ----"),
        call("1"),
        call("x = 100.0 +/- 1.e+00 m del_e = 10.0 +/- 1.e-01 m"),
        call("y = 200.0 +/- 2.e+00 m del_n = 20.0 +/- 2.e-01 m"),
        call("z = 300.0 +/- 3.e+00 m del_u = 30.0 +/- 3.e-01 m"),
        call("Lat. = 1.0 deg, Long. = 2.0, Hgt.msl = 3.0 m"),
        call("2"),
        call("x = 400.0 +/- 2.e+00 m del_e = 40.0 +/- 4.e-01 m"),
        call("y = 500.0 +/- 5.e+00 m del_n = 50.0 +/- 5.e-01 m"),
        call("z = 600.0 +/- 6.e+00 m del_u = 60.0 +/- 6.e-01 m"),
        call("Lat. = 4.0 deg, Long. = 5.0, Hgt.msl = 6.0 m"),
        call("3"),
        call("x = 700.0 +/- 3.e+00 m del_e = 70.0 +/- 7.e-01 m"),
        call("y = 800.0 +/- 8.e+00 m del_n = 80.0 +/- 8.e-01 m"),
        call("z = 900.0 +/- 9.e+00 m del_u = 90.0 +/- 9.e-01 m"),
        call("Lat. = 7.0 deg, Long. = 8.0, Hgt.msl = 9.0 m"),
        call("------------------------"),
        call(),
    ]

    # Go through each call args list and make sure they match to expected
    for idx, arg in enumerate(mock_echo.call_args_list):
        expected = expected_calls[idx]
        assert arg.args == expected.args
