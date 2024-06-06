from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from gnatss.cli import app
from tests import TEST_DATA_FOLDER

runner = CliRunner()


config_yaml_path = (TEST_DATA_FOLDER / "config.yaml").resolve()


@pytest.mark.parametrize(
    "commands",
    [[], ["run"]],
    ids=["root", "run"],
)
def test_app_help(commands):
    result = runner.invoke(app, commands + ["--help"])
    assert result.exit_code == 0


def test_app_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0


def test_module_cli():
    subprocess.run(["python", "-m", "gnatss", "--help"], check=True)


def test_app_run_all_solver_posfilter():
    result = runner.invoke(app, ["run", str(config_yaml_path), "--posfilter", "--solver"])
    with pytest.raises(ValueError):
        raise result.exception


def test_app_run_posfilter():
    result = runner.invoke(app, ["run", str(config_yaml_path), "--posfilter"])
    assert result.exit_code == 0
