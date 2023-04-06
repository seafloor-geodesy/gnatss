import pandas as pd
import pytest
import numpy as np

from . import TEST_DATA_FOLDER
from seagap.harmonic_mean import sv_harmonic_mean, _compute_hm

@pytest.fixture()
def sound_profile_data() -> pd.DataFrame:
    sv_file = TEST_DATA_FOLDER / 'sound_profile_hm.csv'
    return pd.read_csv(sv_file)


@pytest.mark.parametrize(
    "end_depth,expected_hm",
    [
        (-1176.5866, 1481.551),
        (-1146.5881, 1481.521),
        (-1133.7305, 1481.509)
    ]
)
def test_sv_harmonic_mean(end_depth, expected_hm, sound_profile_data):
    svdf = sound_profile_data
    start_depth = -4
    harmonic_mean = round(sv_harmonic_mean(svdf, start_depth, end_depth), 3)

    assert harmonic_mean == expected_hm

@pytest.mark.parametrize(
    "test_idx,expected_hm",
    [
        (3, 1501.69),
        (6, 1501.225)
    ]
)
def test__compute_hm(test_idx, expected_hm):
    dd = np.arange(7) * 10
    sv = np.arange(1502, 1500, step=-0.31)

    # Get partial of the data from test index
    part_dd = dd[:test_idx]
    part_sv = sv[:test_idx]
    start_depth, end_depth = part_dd[0], part_dd[-1]
    result_hm = round(_compute_hm(dd, sv, start_depth, end_depth, 0), 3)
    
    # Check for result to match expected
    assert result_hm == expected_hm

    # Result should be same as regular mean as the data is linear here
    assert result_hm == round(sum(part_sv) / test_idx, 3)
