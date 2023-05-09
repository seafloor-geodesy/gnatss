import pytest
from typer.testing import CliRunner

from gnatss.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "commands",
    [[], ["run"]],
    ids=["root", "run"],
)
def test_app_help(commands):
    result = runner.invoke(app, commands + ["--help"])
    assert result.exit_code == 0

    # TODO: Figure out what kind of test would be good for this
