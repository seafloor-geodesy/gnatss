from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import fsspec


def _get_filesystem(
    input_path: str,
    storage_options: dict[str, Any] = {},  # noqa: B006
) -> Any:
    """Retrieves filesystem from `input_path`"""
    parsed_url = urlparse(input_path)
    return fsspec.filesystem(parsed_url.scheme, **storage_options)


def check_file_exists(
    input_path: str,
    storage_options: dict[str, Any] = {},  # noqa: B006
) -> bool:
    """
    Perform a check if either file or directory exists
    by parsing the `input_path` into a parsed url,
    extracting the scheme, and running a check.

    Parameters
    ----------
    input_path : str
        Input path string. This can be a url path,
        such as `s3://mybucket/file.dat` or
        `s3://mybucket/**/myfile.csv`
    storage_options : dict
        Protocol keyword argument for specified file system.
        This is not needed for local paths.

    Returns
    -------
    bool
        A flag that indicates whether file or directory exists.

    Notes
    -----
    In the case of a glob path string (has ``**``), the function will
    traverse through all child directories until it finds the specified
    file matching the pattern.
    """
    fs = _get_filesystem(input_path, storage_options)
    if "**" in input_path:
        # Check that glob can find files
        # as specified in the input string
        glob_files = fs.glob(input_path)
        return len(glob_files) != 0

    check_path = input_path
    return fs.exists(check_path)


def check_permission(input_path: fsspec.FSMap) -> None:
    """
    Perform permission check to a file directory as specified.
    This function will attempt to write a file called `.permission_test`
    in the base directory.

    Parameters
    ----------
    input_path : fsspec.FSMap
        Input path as mutable mapping

    Returns
    -------
    None

    Raises
    ------
    PermissionError
        If the directory is not writable
    """
    try:
        base_dir = os.path.dirname(input_path.root)  # noqa: PTH120
        if not base_dir:
            base_dir = input_path.root
        TEST_FILE = os.path.join(base_dir, ".permission_test")  # noqa: PTH118
        with input_path.fs.open(TEST_FILE, "w") as f:
            f.write("testing\n")
    except Exception:
        msg = "Writing to specified path is not permitted."
        raise PermissionError(msg) from None
    finally:
        input_path.fs.delete(TEST_FILE)
