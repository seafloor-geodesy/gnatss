"""posfilter.py

The posfilter module containing base models for
position filtering configuration
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .io import InputData


class AtdOffset(BaseModel):
    """Antenna Transducer Offset base model."""

    forward: float = Field(..., description="forward offset")
    rightward: float = Field(..., description="rightward offset")
    downward: float = Field(..., description="downward offset")


class PositionFilterInputs(BaseModel):
    roll_pitch_heading: InputData | None = Field(
        None, description="Roll Pitch Heading (RPH) data path specification."
    )
    novatel: InputData | None = Field(None, description="Novatel data path specification.")
    novatel_std: InputData | None = Field(None, description="Novatel STD data path specification.")
    gps_positions: InputData | None = Field(
        None, description="GPS positions data path specification."
    )


class ExportObs(BaseModel):
    """Export observation data after filtering."""

    full: bool = Field(
        False,
        description="Only export the necessary data, including antenna covariance.",
    )


class PositionFilter(BaseModel):
    """
    Position filter base model for position filtering routine
    """

    export: ExportObs | None = Field(None, description="Export observation data after filtering.")
    input_files: PositionFilterInputs | None = Field(
        None, description="Input files for position filtering routine."
    )
    atd_offsets: AtdOffset = Field(..., description="Antenna Transducer Offset values.")
