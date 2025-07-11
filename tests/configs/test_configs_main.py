from __future__ import annotations

from pathlib import Path

from gnatss.configs.main import Configuration


def test_env_configuration_main(blank_env: None) -> None:
    """Testing a simple configuration class
    with environment variables.

    See root/conftest.py for fixture definition.
    """
    config = Configuration()

    assert config.site_id == "test_site"
