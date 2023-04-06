"""solver.py

The solver module containing base models for
solver configuration
"""
from typing import Any, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, validator

from .io import InputData


class ReferenceEllipsoid(BaseModel):
    """Reference ellipsoid base model"""

    semi_major_axis: float = Field(..., description="Semi-major axis (m)")
    reverse_flattening: float = Field(..., description="Reverse flattening")
    eccentricity: Optional[float] = Field(
        None,
        description="Eccentricity. **This field will be computed during object creation**",
    )

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)

        # Note: Potential improvement with computed value
        # https://github.com/pydantic/pydantic/pull/2625
        __pydantic_self__.eccentricity = (
            2.0 / __pydantic_self__.reverse_flattening
            - (1.0 / __pydantic_self__.reverse_flattening) ** 2.0
        )


class ArrayCenter(BaseModel):
    """Array center base model."""

    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")


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


class SolverGlobal(BaseModel):
    """Solver global base model for inversion process."""

    max_dat = 45000
    max_gps = 423000
    max_del = 15000
    max_brk = 20
    max_surv = 10
    max_sdt_obs = 2000
    max_obm = 472
    max_unmm = 9


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
    # Auto generated uuid per transponder for unique identifier
    _uuid: str = PrivateAttr()

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)

        __pydantic_self__._uuid = uuid4().hex

        # ID is 7 characters based on the uuid
        __pydantic_self__.pxp_id = __pydantic_self__._uuid[:7]


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
        description=(
            "Transducer Delay Time - delay at surface transducer (secs.). "
            "Options: ship/SV3 = 0.0s, WG = 0.1s"
        ),
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
        50.0,
        ge=0.0,
        description=(
            "Maximum residual in centimeters beyond "
            "which data points will be excluded from solution"
        ),
    )

    @validator("transducer_delay_time")
    def check_transducer_delay_time(cls, v):
        """Validate transducer wait time value"""
        if v not in [0.0, 0.1]:
            raise ValueError("Transducer wait time must either be 0.0s or 0.1s")
        return v
