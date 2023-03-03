import os
import fsspec
from urllib.parse import urlparse

def check_file_exists(file_path, storage_options):
    parsed_path = urlparse(file_path)
    fs = fsspec.filesystem(parsed_path.scheme, **storage_options)
    return fs.exists(file_path)


def check_file_permissions(FILE_DIR):
    try:
        base_dir = os.path.dirname(FILE_DIR.root)
        if not base_dir:
            base_dir = FILE_DIR.root
        TEST_FILE = os.path.join(base_dir, ".permission_test").replace("\\", "/")
        with FILE_DIR.fs.open(TEST_FILE, "w") as f:
            f.write("testing\n")
        FILE_DIR.fs.delete(TEST_FILE)
    except Exception:  # noqa
        raise PermissionError("Writing to specified path is not permitted.")