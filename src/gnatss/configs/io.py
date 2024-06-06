from __future__ import annotations

from enum import Enum
from typing import Any

import fsspec
from pydantic import BaseModel, Field, PrivateAttr

from ..utilities.io import _get_filesystem, check_file_exists


class StrEnum(str, Enum):
    """A custom string enum class"""


class CSVOutput(StrEnum):
    """Default CSV output file names"""

    outliers = "outliers.csv"
    residuals = "residuals.csv"
    dist_center = "dist_center.csv"
    deletions = "deletions.csv"
    gps_solution = "gps_solution.csv"


class InputData(BaseModel):
    """Input data path specification base model"""

    path: str = Field(
        ...,
        description="Path string to the data. Ex. s3://bucket/some_data.dat",
    )
    storage_options: dict[str, Any] = Field(
        {},
        description=(
            "Protocol keyword argument for specified file system. "
            "This is not needed for local paths"
        ),
    )
    loader_kwargs: dict[str, Any] = Field(
        {},
        description="Keyword arguments for the data loader.",
    )

    _files: list[str] = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Checks the file
        if not check_file_exists(self.path, self.storage_options):
            msg: str = f"{self.path} doesn't exist!"
            raise FileNotFoundError(msg)

        self._files = self.get_files()

    @property
    def files(self) -> list[str]:
        return self._files

    def get_files(self) -> list[str]:
        """Get the list of files in the directory"""
        fs = _get_filesystem(self.path, self.storage_options)
        return fs.glob(self.path) if "**" in self.path else [self.path]


class OutputPath(BaseModel):
    """Output path base model."""

    path: str = Field(
        ...,
        description="Path string to the output path. Ex. s3://bucket/my_output",
    )
    storage_options: dict[str, Any] = Field(
        {},
        description=(
            "Protocol keyword argument for specified file system. "
            "This is not needed for local paths"
        ),
    )

    _fsmap: str = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        self._fsmap = fsspec.get_mapper(self.path, **self.storage_options)

        # Check to ensure it's a directory
        if not self.path.endswith("/"):
            msg: str = f"{self.path} is not a directory!"
            raise NotADirectoryError(msg)

        # Always try create the directory even if it exists
        self._fsmap.fs.makedirs(self.path, exist_ok=True)

    @property
    def fs(self):
        return self._fsmap.fs
