from pathlib import Path

import pytest

from gnatss.configs.io import InputData
from gnatss.configs.posfilter import PositionFilter, PositionFilterInputs


def test_position_filter_input(blank_csv_test_file: Path) -> None:
    """Testing the PositionFilterInputs class.

    See root/conftest.py for fixture definition.
    """
    test_path = str(blank_csv_test_file)

    # Test initialization of PositionFilter
    inputs = PositionFilterInputs(roll_pitch_heading=InputData(path=test_path))
    pos_filter = PositionFilter(input_files=inputs)
    assert pos_filter.input_files.roll_pitch_heading.path == test_path

    # Test invalid initialization of PositionFilter
    with pytest.raises(Exception) as e:
        pos_filter = PositionFilter()
        assert "1 validation error for PositionFilter\ninput_files\n" in str(e)
