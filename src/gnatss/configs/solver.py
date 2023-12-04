"""solver.py

The solver module containing base models for
solver configuration
"""
from functools import cached_property
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

from .io import InputData


class ReferenceEllipsoid(BaseModel):
    """Reference ellipsoid base model"""

    semi_major_axis: float = Field(..., description="Semi-major axis (m)")
    reverse_flattening: float = Field(..., description="Reverse flattening")

    @computed_field(
        description="Eccentricity. **This field will be computed during object creation**"  # noqa
    )
    @cached_property
    def eccentricity(self) -> Optional[float]:
        return 2.0 / self.reverse_flattening - (1.0 / self.reverse_flattening) ** 2.0


class ArrayCenter(BaseModel):
    """Array center base model."""

    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    alt: float = Field(0.0, description="Altitude")


class SolverInputs(BaseModel):
    sound_speed: InputData = Field(
        ..., description="Sound speed data path specification"
    )
    # NOTE: 3/3/2023 - These are required for now and will change in the future.
    travel_times: InputData = Field(
        ..., description="Travel times data path specification."
    )
    gps_solution: InputData = Field(
        ..., description="GPS solution data path specification."
    )
    deletions: Optional[InputData] = Field(
        None, description="Deletions file for unwanted data points."
    )
    quality_controls: Optional[InputData] = Field(
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
    sv_mean: Optional[float] = Field(
        None, description="Dynamically generated sound velocity mean (m/s)."
    )
    pxp_id: Optional[str] = Field(
        None,
        description=(
            "Transponder id string. "
            "**This field will be computed during object creation**"
        ),
    )
    azimuth: Optional[float] = Field(
        None, description="Transponder azimuth in degrees w.r.t. array center."
    )
    elevation: Optional[float] = Field(
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
    transponders: Optional[List[SolverTransponder]] = Field(
        ..., description="A list of transponders configurations"
    )
    reference_ellipsoid: Optional[ReferenceEllipsoid] = Field(
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
    geoid_undulation: float = Field(
        ..., description="Geoid undulation at sea surface point"
    )
    # TODO: Separate into different plugin for ray tracing
    ray_trace_type: Literal["scale", "1d"] = Field(
        "scale", description="Ray trace method to use"
    )
    bisection_tolerance: float = Field(
        1e-10, description="Tolerance to stop bisection during ray trace"
    )
    array_center: ArrayCenter = Field(
        ..., description="Array center to use for calculation"
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
    input_files: SolverInputs = Field(
        ..., description="Input files data path specifications."
    )
    distance_limit: float = Field(
        150.0,
        ge=0.0,
        description=(
            "Distance in meters from center beyond "
            "which points will be excluded from solution"
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
    travel_times_correction: float = Field(
        0.0, description="Correction to times in travel times (secs.)"
    )
