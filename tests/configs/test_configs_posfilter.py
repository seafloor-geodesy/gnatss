from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gnatss.configs.io import InputData
from gnatss.configs.posfilter import AtdOffset, PositionFilter, PositionFilterInputs


def test_valid_position_filter_input(blank_csv_test_file: Path) -> None:
    """Testing the PositionFilterInputs class.

    See root/conftest.py for fixture definition.
    """
    test_path = str(blank_csv_test_file)

    # Test initialization of PositionFilter
    input_files = PositionFilterInputs(roll_pitch_heading=InputData(path=test_path))
    atdoffsets = AtdOffset(forward=0.0, rightward=-1.2, downward=3.2)

    pos_filter = PositionFilter(input_files=input_files, atd_offsets=atdoffsets)
    assert pos_filter.input_files.roll_pitch_heading.path == test_path
    assert pos_filter.atd_offsets.forward == 0.0
    assert pos_filter.atd_offsets.rightward == -1.2
    assert pos_filter.atd_offsets.downward == 3.2

    pos_filter = PositionFilter(atd_offsets=atdoffsets)
    assert pos_filter.input_files is None
    assert pos_filter.atd_offsets.forward == 0.0
    assert pos_filter.atd_offsets.rightward == -1.2
    assert pos_filter.atd_offsets.downward == 3.2


def test_invalid_position_filter_input(blank_csv_test_file: Path) -> None:
    test_path = str(blank_csv_test_file)
    input_files = PositionFilterInputs(roll_pitch_heading=InputData(path=test_path))

    with pytest.raises(ValidationError) as e:
        _ = PositionFilter()
    assert "1 validation error for PositionFilter\natd_offsets\n" in str(e)

    with pytest.raises(ValidationError) as e:
        _ = PositionFilter(input_files=input_files)
    assert "1 validation error for PositionFilter\natd_offsets\n" in str(e)
