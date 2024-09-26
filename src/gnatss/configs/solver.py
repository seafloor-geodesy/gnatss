"""solver.py

The solver module containing base models for
solver configuration
"""

from __future__ import annotations

from functools import cached_property
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

from .io import InputData
from .transponders import Transponder


class ReferenceEllipsoid(BaseModel):
    """Reference ellipsoid base model"""

    semi_major_axis: float = Field(..., description="Semi-major axis (m)")
    reverse_flattening: float = Field(..., description="Reverse flattening")

    @computed_field(
        description="Eccentricity. **This field will be computed during object creation**"
    )
    @cached_property
    def eccentricity(self) -> float | None:
        return 2.0 / self.reverse_flattening - (1.0 / self.reverse_flattening) ** 2.0


class ArrayCenter(BaseModel):
    """Array center base model."""

    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    alt: float = Field(0.0, description="Altitude")


class GPSSolutionInput(InputData):
    legacy: bool = Field(
        False, description="Flag to indicate if the input file is in legacy format."
    )


class SolverInputs(BaseModel):
    sound_speed: InputData | None = Field(None, description="Sound speed data path specification")
    # NOTE: 3/3/2023 - These are required for now and will change in the future.
    travel_times: InputData | None = Field(
        None, description="Travel times data path specification."
    )
    gps_solution: GPSSolutionInput | None = Field(
        None, description="GPS solution data path specification."
    )
    deletions: InputData | None = Field(
        None, description="Deletions file for unwanted data points."
    )
    quality_controls: InputData | None = Field(
        None,
        description="Quality control file(s) for user defined unwanted data points.",
    )


class SolverGlobal(BaseModel):
    """Solver global base model for inversion process."""

    max_dat: int = 45000
    max_gps: int = 423000
    max_del: int = 15000
    max_brk: int = 20
    max_surv: int = 10
    max_sdt_obs: int = 2000
    max_obm: int = 472
    max_unmm: int = 9


class SolverTransponder(BaseModel):
    """Solver transponder base model for each transponder configuration"""

    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    height: float = Field(..., description="Transponder depth in meters (m).")
    internal_delay: float = Field(
        ...,
        description=(
            "Transponder internal delay in seconds (s). "
            "Assume transponder delay contains: "
            "delay-line, non-delay-line internal delays "
            "(determined at transdec) and any user set delay (dail-in)."
        ),
    )
    sv_mean: float | None = Field(
        None, description="Dynamically generated sound velocity mean (m/s)."
    )
    pxp_id: str | None = Field(
        None,
        description=(
            "Transponder id string. " "**This field will be computed during object creation**"
        ),
    )
    azimuth: float | None = Field(
        None, description="Transponder azimuth in degrees w.r.t. array center."
    )
    elevation: float | None = Field(
        None, description="Transponder elevation in degrees w.r.t. array center."
    )

    @computed_field(repr=False, description="Transponder unique identifier")
    @cached_property
    def _uuid(self) -> str:
        """Auto generated uuid per transponder for unique identifier"""
        return uuid4().hex


class Solver(BaseModel):
    """
    Solver transponder base model containing configurations
    for GNSS-A inversion computation
    """

    defaults: SolverGlobal = SolverGlobal()
    transponders: list[Transponder] | None = Field(
        None, description="A list of transponders configurations"
    )
    reference_ellipsoid: ReferenceEllipsoid | None = Field(
        ..., description="Reference ellipsoid configurations"
    )
    gps_sigma_limit: float = Field(
        0.05,
        description="Maximum positional sigma allowed to use GPS positions",
        gt=0.0,
        lt=100.0,
    )
    std_dev: bool = Field(
        True,
        description="GPS positional uncertainty flag std. dev. (True) or variance (False)",
    )
    geoid_undulation: float = Field(..., description="Geoid undulation at sea surface point")
    # TODO: Separate into different plugin for ray tracing
    ray_trace_type: Literal["scale", "1d"] = Field("scale", description="Ray trace method to use")
    bisection_tolerance: float = Field(
        1e-10, description="Tolerance to stop bisection during ray trace"
    )
    array_center: ArrayCenter | None = Field(
        None, description="Array center to use for calculation"
    )
    travel_times_variance: float = Field(
        1e-10, description="VARIANCE (s**2) PXP two-way travel time measurement"
    )
    transducer_delay_time: float = Field(
        0.0,
        description="Transducer Delay Time - delay at surface transducer (secs). ",
    )
    harmonic_mean_start_depth: float = Field(
        0.0, description="Start depth in meters for harmonic mean computation."
    )
    input_files: SolverInputs = Field(..., description="Input files data path specifications.")
    distance_limit: float = Field(
        150.0,
        ge=0.0,
        description=(
            "Distance in meters from center beyond " "which points will be excluded from solution"
        ),
    )
    residual_limit: float = Field(
        10000.0,
        ge=0.0,
        description=(
            "Maximum residual in centimeters beyond "
            "which data points will be excluded from solution"
        ),
    )
    residual_range_limit: float = Field(
        20000.0,
        ge=0.0,
        description=(
            "Maximum residual range (maximum - minimum) "
            "in centimeters for a given epoch, beyond which "
            "data points will be excluded from solution"
        ),
    )
    residual_outliers_threshold: float = Field(
        25.0,
        ge=0.0,
        description="Residual outliers threshold acceptable before throwing an error in percent",
    )
    travel_times_correction: float = Field(
        0.0, description="Correction to times in travel times (secs.)"
    )
    twtt_model: Literal["simple_twtt"] = Field(
        "simple_twtt", description="Travel time model to use."
    )
