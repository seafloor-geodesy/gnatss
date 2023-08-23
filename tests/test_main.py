from gnatss.main import gather_files


def test_gather_files(mocker):
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
    item_keys = ["sound_speed", "travel_times", "gps_solution", "deletions"]
    sample_dict = {
        k: {
            "path": f"/some/path/to/{k}"
            if k != "travel_times"
            else "/some/glob/**/path",
            "storage_options": {},
        }
        for k in item_keys
    }
    config = mocker.patch("gnatss.configs.main.Configuration")
    config.solver.input_files.model_dump.return_value = sample_dict

    # Perform test
    all_files_dict = gather_files(config)
    # Check all_files_dict
    assert isinstance(all_files_dict, dict)
    assert sorted(list(all_files_dict.keys())) == sorted(item_keys)

    # Test glob
    assert isinstance(all_files_dict["travel_times"], list)
    assert all_files_dict["travel_times"] == glob_res
