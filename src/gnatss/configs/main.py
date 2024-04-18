"""main.py

The main configuration module containing base settings pydantic
classes
"""
import datetime
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

import numpy as np
import pymap3d
import yaml
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ..utilities.geo import ecef2ae
from .io import InputData, OutputPath
from .posfilter import PositionFilter
from .solver import Solver
from .transponders import Transponder

CONFIG_FILE = "config.yaml"


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source that reads from a yaml file.

    Read config settings form a local yaml file where the software runs
    """

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """
        Gets the value,
        the key for model creation,
        and a flag to determine whether value is complex.

        *This is an override for the pydantic abstract method.*
        """
        encoding = self.config.get("env_file_encoding")
        config_path = Path(CONFIG_FILE)
        file_content_yaml = {}
        if config_path.exists():
            # Only load config.yaml when it exists
            file_content_yaml = yaml.safe_load(config_path.read_text(encoding))
        else:
            # Warn user when config.yaml is not found
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
        """
        Prepares the value of a field.

        *This is an override for the pydantic abstract method.*
        """
        return value

    def __call__(self) -> Dict[str, Any]:
        """
        Allows the class to be called as a function.
        """
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


class ArrayCenter(BaseModel):
    """Array center base model."""

    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    alt: float = Field(0.0, description="Altitude")


class MainInputs(BaseModel):
    travel_times: Optional[InputData] = Field(
        None, description="Input travel times data path specification"
    )


class Configuration(BaseConfiguration):
    """Configuration class to generate config object"""

    # General configurations
    site_id: str = Field(..., description="GNSS-A site name or code")
    campaign: Optional[str] = Field(None, description="Observation campaign name")
    time_origin: Optional["str | datetime.datetime"] = Field(
        None, description="Origin of time used in the file [UTC]"
    )
    ref_frame: Literal["wgs84"] = Field(
        "wgs84", description="Reference frame used in the file"
    )
    array_center: ArrayCenter = Field(
        ..., description="Array center to use for calculation"
    )
    transponders: List[Transponder] = Field(
        ..., description="A list of transponders configurations"
    )
    travel_times_variance: float = Field(
        1e-10, description="VARIANCE (s**2) PXP two-way travel time measurement"
    )
    travel_times_correction: float = Field(
        0.0, description="Correction to times in travel times (secs.)"
    )
    transducer_delay_time: float = Field(
        0.0,
        description="Transducer Delay Time - delay at surface transducer (secs). ",
    )

    # Processing configurations
    solver: Optional[Solver] = Field(None, description="Solver configurations")
    posfilter: Optional[PositionFilter] = Field(
        None, description="Position filter configurations"
    )

    # File related configurations
    input_files: MainInputs = Field(
        ..., description="Input files data path specifications."
    )
    output: Optional[OutputPath] = Field(None, description="Output path configurations")

    # Extra configurations
    notes: Optional[str] = Field(None, description="Any other optional comments")

    def __init__(self, **data):
        super().__init__(**data)

        self.setup_transponders()

        # Set solver configurations
        if self.solver is not None:
            self.solver.array_center = self.array_center
            self.solver.transponders = self.transponders
            self.solver.travel_times_variance = self.travel_times_variance
            self.solver.transducer_delay_time = self.transducer_delay_time
            self.solver.travel_times_correction = self.travel_times_correction

    def setup_transponders(self):
        # Set the transponders pxp id based on the site id
        transponders = self.transponders

        if self.array_center is None:
            raise ValueError("Array center is not set")

        for idx in range(len(transponders)):
            # Compute azimuth and elevation
            tp = transponders[idx]
            arr_center = self.array_center

            # Convert geodetic (deg) to ecef (meters)
            x, y, z = pymap3d.geodetic2ecef(tp.lat, tp.lon, tp.alt)

            # Compute azimuth and elevation w.r.t. array center
            az, el = ecef2ae(x, y, z, arr_center.lat, arr_center.lon, arr_center.alt)

            # Round the values to 2 decimal places and
            # take absolute value of elevation
            az, el = np.round([az, np.abs(el)], 2)

            # Set azimuth and elevation
            transponders[idx].azimuth = az
            transponders[idx].elevation = el

            # Set pxp id, if not set
            if transponders[idx].pxp_id is None:
                transponders[idx].pxp_id = "-".join([self.site_id, str(idx + 1)])
