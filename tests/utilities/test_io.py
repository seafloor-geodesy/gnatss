import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp

import fsspec
import pytest

from gnatss.utilities.io import check_file_exists, check_permission

PREFIX = "gnatss-"
NON_EXISTENT = "non_existent"


@pytest.fixture(params=["single", "glob", "random", "no_files_found"])
def input_path(request):
    # TODO: Add remote file test
    _, tmp_file = mkstemp(prefix=PREFIX)
    file_path = Path(tmp_file)
    if request.param == "single":
        file_path = str(file_path)
    elif request.param == "glob":
        # Get file directory 2 levels above
        file_dir = file_path.parent
        dir_path = file_dir.parent / "**" / file_path.name
        file_path = str(dir_path)
    elif request.param == "random":
        file_path = f"./{NON_EXISTENT}"
    elif request.param == "no_files_found":
        # Create a unique directory path that does not exist
        file_path = f"./non_existent_directory/**/{NON_EXISTENT}"

    yield file_path

    # Clean up after
    if os.path.exists(file_path):
        os.unlink(file_path)


def test_check_file_exists(input_path):
    """Tests the `check_file_exists` io function"""
    # A non existent file ... only path string
    expected_value = NON_EXISTENT not in input_path

    assert isinstance(check_file_exists(input_path), bool) is True
    assert check_file_exists(input_path) is expected_value


@pytest.mark.parametrize("has_permission", [True, False])
def test_check_permission(has_permission):
    """Tests the `check_permission` io function"""
    # TODO: Add remote file test
    with TemporaryDirectory(prefix=PREFIX) as temp_path:
        if not has_permission:
            # Change directory permission to read only
            os.chmod(temp_path, stat.S_IREAD)

        fmap = fsspec.get_mapper(temp_path)
        try:
            # This should pass when there's permission
            # otherwise check that during a PermissionError
            # `has_permission` is False
            check_permission(input_path=fmap)
        except PermissionError:
            assert has_permission is False
