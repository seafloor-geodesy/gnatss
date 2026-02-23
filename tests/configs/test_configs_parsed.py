from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gnatss.configs.io import InputData
from gnatss.configs.parsed import Parsed, ParsedInputs
from gnatss.configs.posfilter import AtdOffset


def test_valid_parsed_qc_input(blank_csv_test_file: Path) -> None:
    """Testing the ParsedInputs class.

    See root/conftest.py for fixture definition.
    """
    test_path = str(blank_csv_test_file)
    test_format = "targz"

    # Test initialization of PositionFilter
    input_files = ParsedInputs(raw_data=InputData(path=test_path,format=test_format))
    atdoffsets = AtdOffset(forward=0.0, rightward=-1.2, downward=3.2)

    pos_filter = Parsed(input_files=input_files, atd_offsets=atdoffsets)
    assert pos_filter.input_files.raw_data.path == test_path
    assert pos_filter.input_files.raw_data.format == test_format
    assert pos_filter.atd_offsets.forward == 0.0
    assert pos_filter.atd_offsets.rightward == -1.2
    assert pos_filter.atd_offsets.downward == 3.2

    pos_filter = Parsed(atd_offsets=atdoffsets)
    assert pos_filter.input_files is None
    assert pos_filter.atd_offsets.forward == 0.0
    assert pos_filter.atd_offsets.rightward == -1.2
    assert pos_filter.atd_offsets.downward == 3.2


def test_invalid_parsed_qc_input(blank_csv_test_file: Path) -> None:
    test_path = str(blank_csv_test_file)
    input_files = ParsedInputs(raw_data=InputData(path=test_path))

    with pytest.raises(ValidationError) as e:
        _ = Parsed()
    assert "1 validation error for Parsed\natd_offsets\n" in str(e)

    with pytest.raises(ValidationError) as e:
        _ = Parsed(input_files=input_files)
    assert "1 validation error for Parsed\natd_offsets\n" in str(e)
