from __future__ import annotations

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.options.sessions = ["lint", "tests", "build"]


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


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
