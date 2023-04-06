import pytest
from typer.testing import CliRunner

from seagap.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "commands,expected_stdouts",
    [
        (
            [],
            [
                "GNSS-A Processing in Python",
                "run    Run the full pre-processing routine for GNSS-A",
            ],
        ),
        (
            ["run"],
            [
                "Run the full pre-processing routine for GNSS-A",
                "--config-yaml",
                "Custom path to configuration yaml file.",
            ],
        ),
    ],
    ids=["root", "run"],
)
def test_app_help(commands, expected_stdouts):
    result = runner.invoke(app, commands + ["--help"])
    assert result.exit_code == 0

    for expected in expected_stdouts:
        assert expected in result.stdout
