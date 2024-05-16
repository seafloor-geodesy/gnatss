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
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


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
    test_data_zip_path = (DIR / "tests" / "data" / "2022.zip").absolute()
    session.install(".[test]")
    session.run("git", "lfs", "pull", external=True)
    session.run(
        "unzip",
        str(test_data_zip_path),
        "-d",
        str(test_data_zip_path.parent),
        external=True,
    )
    session.run("pytest", "-vvv", "tests", *session.posargs)


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
