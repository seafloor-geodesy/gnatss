"""posfilter.py

The posfilter module containing base models for
position filtering configuration
"""

from pydantic import BaseModel, Field

from .io import InputData


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
