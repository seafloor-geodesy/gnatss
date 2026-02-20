"""parsed.py

The parsed module containing base models for
position filtering configuration when only 
2-minute parsed data is available, generally
during data collection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .io import InputData


class AtdOffset(BaseModel):
    """Antenna Transducer Offset base model."""

    forward: float = Field(..., description="forward offset")
    rightward: float = Field(..., description="rightward offset")
    downward: float = Field(..., description="downward offset")


class ParsedInputs(BaseModel):
    gps_positions: InputData | None = Field(
        None, description="GPS positions data path specification."
    )
    raw_data: InputData | None = Field(
        None, description="Raw data files telemetered to shore (Sonardyne SV-3 format, *.tar.gz)"
    )


class ExportObs(BaseModel):
    """Export observation data after filtering."""

    full: bool = Field(
        False,
        description="Only export the necessary data, including antenna covariance.",
    )


class Parsed(BaseModel):
    """
    Position filter base model for position filtering routine
    """

    export: ExportObs | None = Field(None, description="Export observation data after filtering.")
    input_files: ParsedInputs | None = Field(
        None, description="Input files for 2-minute parsed position filtering routine."
    )
    atd_offsets: AtdOffset = Field(..., description="Antenna Transducer Offset values.")
