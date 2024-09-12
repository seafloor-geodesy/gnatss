from __future__ import annotations

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.options.sessions = ["lint", "tests", "build", "docs"]


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """
    Setup and Build the documentation.
    """
    setup_docs(session)
    build_docs(session)


@nox.session
def setup_docs(session: nox.Session) -> None:
    """
    Setup the documentation.
    """
    excluded_api_modules = [
        "src/gnatss/version.py",
        "src/gnatss/configs",
        "src/gnatss/dataspec/v1.py",
    ]
    session.install(".[all]")

    # Generate the API documentation
    session.run(
        "sphinx-apidoc",
        "-f",
        "-M",
        "-e",
        "-T",
        "--implicit-namespaces",
        "-o",
        "docs/api/",
        "src/gnatss/",
        *excluded_api_modules,
    )

    # Setup the sphinx conf file from _config.yml
    session.run("jb", "config", "sphinx", "docs")


@nox.session
def build_docs(session: nox.Session) -> None:
    """
    Build the documentation. This session depends on the `setup_docs` session.
    """
    session.run("jb", "build", "./docs")


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.

    Notes
    -----
    This session will download the test data from the repository
    from git lfs. Must have the following installed in your system:
    - git-lfs: https://github.com/git-lfs/git-lfs
    - unzip: https://linuxize.com/post/how-to-unzip-files-in-linux/
    """
    # Specify the setuptools version
    # https://numpy.org/doc/stable/reference/distutils_status_migration.html#distutils-status-migration
    session.install("setuptools<60")
    session.install(".[test]")
    if not (DIR / "tests" / "data" / "2022").exists():
        session.run(
            "python",
            "-c",
            "from gnatss.utilities.testing import download_test_data; download_test_data(unzip=True)",
        )
    with session.chdir("tests/fortran"):
        # Runs in the tests fortran directory
        session.run("f2py", "-c", "-m", "flib", "xyz2enu.f")

    # Run in original directory
    session.run("pytest", *session.posargs)


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR / "dist"
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
