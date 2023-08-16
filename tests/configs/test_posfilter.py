from pathlib import Path

import pytest

from gnatss.configs.io import InputData
from gnatss.configs.posfilter import PositionFilter, PositionFilterInputs


def test_position_filter_input():
    # Create test file
    test_file = Path(__file__).parent / "test_data.csv"
    test_file.touch()

    # Test initialization of PositionFilter
    inputs = PositionFilterInputs(roll_pitch_heading=InputData(path=str(test_file)))
    pos_filter = PositionFilter(input_files=inputs)
    assert pos_filter.input_files.roll_pitch_heading.path == str(test_file)

    # Delete test file
    test_file.unlink()

    # Test invalid initialization of PositionFilter
    with pytest.raises(Exception) as e:
        pos_filter = PositionFilter()
        assert "1 validation error for PositionFilter\ninput_files\n" in str(e)
