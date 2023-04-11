import pytest
from typer.testing import CliRunner

from seagap.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "commands,expected_stdouts_len",
    [
        (
            [],
            16,
        ),
        (
            ["run"],
            13,
        ),
    ],
    ids=["root", "run"],
)
def test_app_help(commands, expected_stdouts_len):
    result = runner.invoke(app, commands + ["--help"])
    assert result.exit_code == 0

    # TODO: Expand this to get a listing of expected sub commands and check against those
    assert len(result.stdout.split("\n")) == expected_stdouts_len
