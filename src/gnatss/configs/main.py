"""main.py

The main configuration module containing base settings pydantic
classes
"""

from __future__ import annotations

import datetime
from typing import Literal

import numpy as np
import pymap3d
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..utilities.geo import ecef2ae
from .io import InputData, OutputPath
from .posfilter import PositionFilter
from .solver import Solver
from .transponders import Transponder


class BaseConfiguration(BaseSettings):
    """Base configuration class"""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="gnatss_",
    )


class ArrayCenter(BaseModel):
    """Array center base model."""

    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    alt: float = Field(0.0, description="Altitude")


class MainInputs(BaseModel):
    travel_times: InputData | None = Field(
        None, description="Input travel times data path specification"
    )


class Configuration(BaseConfiguration):
    """Configuration class to generate config object"""

    # General configurations
    site_id: str = Field(..., description="GNSS-A site name or code")
    campaign: str | None = Field(None, description="Observation campaign name")
    time_origin: str | datetime.datetime | None = Field(
        None, description="Origin of time used in the file [UTC]"
    )
    ref_frame: Literal["wgs84"] = Field("wgs84", description="Reference frame used in the file")
    array_center: ArrayCenter = Field(..., description="Array center to use for calculation")
    transponders: list[Transponder] = Field(
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
    solver: Solver | None = Field(None, description="Solver configurations")
    posfilter: PositionFilter | None = Field(None, description="Position filter configurations")

    # File related configurations
    input_files: MainInputs = Field(..., description="Input files data path specifications.")
    output: OutputPath | None = Field(None, description="Output path configurations")

    # Extra configurations
    notes: str | None = Field(None, description="Any other optional comments")

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
            msg = "Array center is not set"
            raise ValueError(msg)

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
