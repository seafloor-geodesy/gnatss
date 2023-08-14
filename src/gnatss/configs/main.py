"""main.py

The main configuration module containing base settings pydantic
classes
"""
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .io import OutputPath
from .solver import Solver

CONFIG_FILE = "config.yaml"


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source that reads from a yaml file.

    Read config settings form a local yaml file where the software runs

    """

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        encoding = self.config.get("env_file_encoding")
        config_path = Path(CONFIG_FILE)
        file_content_yaml = {}
        if config_path.exists():
            # Only load config.yaml when it exists
            file_content_yaml = yaml.safe_load(config_path.read_text(encoding))
        else:
            warnings.warn(
                (
                    f"Configuration file `{CONFIG_FILE}` not found. "
                    "Will attempt to retrieve configuration from environment variables."
                )
            )

        field_value = file_content_yaml.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value

        return d


def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    Read config settings form a local yaml file where the software runs

    Parameters
    ----------
    settings : pydantic.BaseSettings
        The base settings class

    Returns
    -------
    dict
        The configuration dictionary based on inputs from the yaml
        file
    """
    encoding = settings.__config__.env_file_encoding
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        # Only load config.yaml when it exists
        return yaml.safe_load(config_path.read_text(encoding))
    else:
        warnings.warn(
            (
                f"Configuration file `{CONFIG_FILE}` not found. "
                "Will attempt to retrieve configuration from environment variables."
            )
        )
    return {}


class BaseConfiguration(BaseSettings):
    """Base configuration class"""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="gnatss_",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            file_secret_settings,
        )


class Configuration(BaseConfiguration):
    """Configuration class to generate config object"""

    site_id: str = Field(..., description="The site identification for processing")
    # TODO: Separate settings out to core plugin
    solver: Optional[Solver] = Field(None, description="Solver configurations")
    output: OutputPath

    def __init__(self, **data):
        super().__init__(**data)

        # Set the transponders pxp id based on the site id
        transponders = self.solver.transponders
        for idx in range(len(transponders)):
            transponders[idx].pxp_id = "-".join([self.site_id, str(idx + 1)])
