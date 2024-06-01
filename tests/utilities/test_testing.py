import pytest
from pathlib import Path
from gnatss.utilities.testing import download_test_data


@pytest.mark.parametrize("unzip", [True, False])
def test_download_test_data(tmp_path, unzip):
    test_data_zip_file = tmp_path / "2022.zip"
    returned_path = download_test_data(str(test_data_zip_file), unzip=unzip)

    assert isinstance(returned_path, str)
    assert Path(returned_path).exists()

    if unzip:
        assert Path(returned_path) == tmp_path
    else:
        assert Path(returned_path).parent == tmp_path
