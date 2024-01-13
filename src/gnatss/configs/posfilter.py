"""posfilter.py

The posfilter module containing base models for
position filtering configuration
"""

from pydantic import BaseModel, Field

from .io import InputData


class AtdOffset(BaseModel):
    """Antenna Transducer Offset base model."""

    forward: float = Field(..., description="forward offset")
    rightward: float = Field(..., description="rightward offset")
    downward: float = Field(..., description="downward offset")


class PositionFilterInputs(BaseModel):
    roll_pitch_heading: InputData = Field(
        ..., description="Roll Pitch Heading (RPH) data path specification."
    )


class PositionFilter(BaseModel):
    """
    Position filter base model for position filtering routine
    """

    input_files: PositionFilterInputs = Field(
        ..., description="Input files for position filtering routine."
    )
    atd_offsets: AtdOffset = Field(..., description="Antenna Transducer Offset values.")
