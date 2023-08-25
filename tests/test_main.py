import pytest

from gnatss.main import gather_files


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
        getattr(config, proc).input_files.dict.return_value = sample_dict

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
