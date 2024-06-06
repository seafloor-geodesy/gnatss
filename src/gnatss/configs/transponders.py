from __future__ import annotations

import warnings
from functools import cached_property
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


class Transponder(BaseModel):
    """Transponder base model for each transponder configuration"""

    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    height: float = Field(..., description="Transponder height in meters (m).")
    alt: float = Field(..., description="Transponder depth in meters (m).")
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

    @model_validator(mode="before")
    @classmethod
    def sync_height_alt(cls, data: Any) -> Any:
        """Sync height and alt values"""
        if isinstance(data, dict):
            if data.get("height") is not None:
                warnings.warn(
                    "Use 'alt' instead of 'height' for transponder depth.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data["alt"] = data["height"]
            elif data.get("alt") is not None:
                data["height"] = data["alt"]

        return data
