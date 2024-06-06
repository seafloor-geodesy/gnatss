from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlretrieve

try:
    from github import Github
except ImportError:
    raise ImportError("Please install PyGithub: pip install PyGithub")

TEST_DATA_REPO = "seafloor-geodesy/gnatss-test-data"

# Open read-only authentication to Github
gh = Github()
PWD = Path()
TEST_DATA_ZIP_FILE = (PWD / "tests" / "data" / "2022.zip").resolve()


def download_test_data(zip_file_path: str | None = None, unzip: bool = False) -> str:
    if zip_file_path is not None:
        global TEST_DATA_ZIP_FILE  # noqa: PLW0603
        TEST_DATA_ZIP_FILE = Path(zip_file_path).resolve()

    repo = gh.get_repo(TEST_DATA_REPO)
    release = repo.get_latest_release()
    asset = next(asset for asset in release.get_assets())
    url = asset.browser_download_url
    print(f"Downloading test data from {url}")
    path, _ = urlretrieve(url, str(TEST_DATA_ZIP_FILE))
    print(f"Downloaded test data to {path}")

    if unzip:
        test_data_dir = TEST_DATA_ZIP_FILE.parent
        print(f"Unzipping {path} to {test_data_dir}")
        test_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(path, test_data_dir, "zip")
        print(f"Unzipped {path} to {test_data_dir}")
        return str(test_data_dir)
    return path
