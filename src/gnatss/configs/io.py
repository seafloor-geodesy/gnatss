from enum import Enum
from typing import Any, Dict

import fsspec
from pydantic import BaseModel, Field, PrivateAttr

from ..utilities.io import check_file_exists, check_permission


class StrEnum(str, Enum):
    """A custom string enum class"""

    ...


class CSVOutput(StrEnum):
    """Default CSV output file names"""

    outliers = "outliers.csv"
    residuals = "residuals.csv"
    dist_center = "dist_center.csv"


class InputData(BaseModel):
    """Input data path specification base model"""

    path: str = Field(
        ...,
        description="Path string to the data. Ex. s3://bucket/some_data.dat",
    )
    storage_options: Dict[str, Any] = Field(
        {},
        description=(
            "Protocol keyword argument for specified file system. "
            "This is not needed for local paths"
        ),
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Checks the file
        if not check_file_exists(self.path, self.storage_options):
            raise FileNotFoundError("The specified file doesn't exist!")


class OutputPath(BaseModel):
    """Output path base model."""

    path: str = Field(
        ...,
        description="Path string to the output path. Ex. s3://bucket/my_output",
    )
    storage_options: Dict[str, Any] = Field(
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
        # Checks the file permission as the object is being created
        check_permission(self._fsmap)
