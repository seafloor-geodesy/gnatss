"""main.py

The main configuration module containing base settings pydantic
classes
"""
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseSettings, Field
from pydantic.fields import ModelField

from .solver import Solver

CONFIG_FILE = "config.yaml"


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

    @classmethod
    def add_fields(cls, **field_definitions: Any) -> None:
        """
        Adds additional configuration field on the fly (inplace)

        Parameters
        ----------
        **field_definitions
            Keyword arguments of the new field to be added
        """
        new_fields: Dict[str, ModelField] = {}
        new_annotations: Dict[str, Optional[type]] = {}

        for f_name, f_def in field_definitions.items():
            if isinstance(f_def, tuple):
                try:
                    f_annotation, f_value = f_def
                except ValueError as e:
                    raise Exception(
                        "field definitions should either be a tuple of"
                        " (<type>, <default>) or just a "
                        "default value, unfortunately this means tuples as "
                        "default values are not allowed"
                    ) from e
            else:
                f_annotation, f_value = None, f_def

            if f_annotation:
                new_annotations[f_name] = f_annotation

            new_fields[f_name] = ModelField.infer(
                name=f_name,
                value=f_value,
                annotation=f_annotation,
                class_validators=None,
                config=cls.__config__,
            )

        cls.__fields__.update(new_fields)
        cls.__annotations__.update(new_annotations)

    class Config:
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        env_prefix = "seagap_"

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                yaml_config_settings_source,
                env_settings,
                file_secret_settings,
            )


class Configuration(BaseConfiguration):
    """Configuration class to generate config object"""

    site_id: str = Field(..., description="The site identification for processing")
    # TODO: Separate settings out to core plugin
    solver: Optional[Solver] = Field(None, description="Solver configurations")
