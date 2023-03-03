from tempfile import mkstemp, mkdtemp
from pathlib import Path
import stat
import os

import fsspec
import pytest

from seagap.utilities.io import check_file_exists, check_permission

PREFIX = "seagap-"

@pytest.fixture(params=["single", "glob", "random"])
def input_path(request):
    # TODO: Add remote file test
    _, tmp_file = mkstemp(prefix=PREFIX)
    file_path = Path(tmp_file)
    file_dir = file_path.parent
    if request.param == "single":
        return file_path, str(file_path)
    elif request.param == "glob":
        # Get file directory 2 level above
        dir_path = str(file_dir.parent)
        return file_path, f"{dir_path}/**/{file_path.name}"
    elif request.param == "random":
        return "./non_existent"


def test_check_file_exists(input_path):
    """Tests the `check_file_exists` io function"""
    file_path = None
    if isinstance(input_path, str):
        # A non existent file ... only path string
        expected_value = False
        test_path = input_path
    else:
        expected_value = True
        file_path, test_path = input_path

    assert isinstance(check_file_exists(test_path), bool) is True
    assert check_file_exists(test_path) is expected_value
    
    # Clean up after
    if file_path is not None:
        os.unlink(file_path)

@pytest.mark.parametrize("has_permission", [True, False])
def test_check_permission(has_permission):
    """Tests the `check_permission` io function"""
    # TODO: Add remote file test
    temp_path = mkdtemp(prefix=PREFIX)
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
