from typing import Any, Dict, List, Literal, Optional

import fsspec

from pydantic import BaseModel, Field, PrivateAttr, validator

from ..utilities.io import check_file_exists, check_permission

class InputData(BaseModel):
    """Input data path specification base model"""

    path: str = Field(
        ...,
        description="Path string to the data. Ex. s3://bucket/some_data.dat",
    )
    storage_options: Dict[str, Any] = Field(
        {},
        description="""Protocol keyword argument for specified file system.
        This is not needed for local paths""",
    )

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)

        # Checks the file
        if not check_file_exists(
            __pydantic_self__.path, __pydantic_self__.storage_options
        ):
            raise FileNotFoundError("The specified file doesn't exist!")
        
class OutputPath(BaseModel):
    """Output path base model."""

    path: str = Field(
        ...,
        description="Path string to the output path. Ex. s3://bucket/my_output",
    )
    storage_options: Dict[str, Any] = Field(
        {},
        description="""Protocol keyword argument for specified file system.
        This is not needed for local paths""",
    )

    _fsmap: str = PrivateAttr()

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)

        __pydantic_self__._fsmap = fsspec.get_mapper(
            __pydantic_self__.path, **__pydantic_self__.storage_options
        )
        # Checks the file permission as the object is being created
        check_permission(__pydantic_self__._fsmap)
