from unittest.mock import call

import pytest

from gnatss.configs.solver import SolverTransponder
from gnatss.main import _get_latest_process, _print_final_stats, gather_files


@pytest.mark.parametrize("proc", ["solver", "posfilter", "random"])
def test_gather_files(mocker, proc):
    tt = "travel_times"
    rph = "roll_pitch_heading"
    glob_vals = [tt, rph]
    expected_procs = {
        "solver": ["sound_speed", tt, "gps_solution", "deletions"],
        "posfilter": [rph],
    }

    # Setup get_filesystem mock
    glob_res = [
        "/some/path/to/1",
        "/some/path/to/2",
        "/some/path/to/3",
    ]

    class Filesystem:
        def glob(path):
            return glob_res

    mocker.patch("gnatss.main._get_filesystem", return_value=Filesystem)

    # Setup mock configuration
    item_keys = []
    if proc in expected_procs:
        item_keys = expected_procs[proc]

    sample_dict = {
        k: {
            "path": f"/some/path/to/{k}"
            if k not in glob_vals
            else "/some/glob/**/path",
            "storage_options": {},
        }
        for k in item_keys
    }
    config = mocker.patch("gnatss.configs.main.Configuration")
    if proc in list(expected_procs.keys()):
        # Test for actual proc that exists
        getattr(config, proc).input_files.model_dump.return_value = sample_dict

        all_files_dict = gather_files(config, proc=proc)
        # Check all_files_dict
        assert isinstance(all_files_dict, dict)
        assert sorted(list(all_files_dict.keys())) == sorted(item_keys)

        # Test glob
        for val in glob_vals:
            if val in all_files_dict:
                assert isinstance(all_files_dict[val], list)
                assert all_files_dict[val] == glob_res
    else:
        # Test for random
        del config.random

        with pytest.raises(AttributeError) as exc_info:
            all_files_dict = gather_files(config, proc=proc)

        assert exc_info.type == AttributeError
        assert exc_info.value.args[0] == f"Unknown process type: {proc}"


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
        SolverTransponder(
            pxp_id="1", lat=1.1, lon=2.1, height=3.1, internal_delay=0.01
        ),
        SolverTransponder(
            pxp_id="2", lat=1.2, lon=2.2, height=3.2, internal_delay=0.02
        ),
        SolverTransponder(
            pxp_id="3", lat=1.3, lon=2.3, height=3.3, internal_delay=0.03
        ),
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
