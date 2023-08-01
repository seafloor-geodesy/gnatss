import numpy as np
import pandas as pd
import pytest

from gnatss.constants import SP_DEPTH, SP_SOUND_SPEED
from gnatss.harmonic_mean import _compute_hm, sv_harmonic_mean

from . import TEST_DATA_FOLDER


@pytest.fixture()
def sound_profile_data() -> pd.DataFrame:
    sv_file = TEST_DATA_FOLDER / "sound_profile_hm.csv"
    return pd.read_csv(sv_file)


@pytest.mark.parametrize(
    "end_depth,expected_hm",
    [(-1176.5866, 1481.542), (-1146.5881, 1481.513), (-1133.7305, 1481.5)],
)
def test_sv_harmonic_mean(end_depth, expected_hm, sound_profile_data):
    svdf = sound_profile_data
    start_depth = -4
    harmonic_mean = round(sv_harmonic_mean(svdf, start_depth, end_depth), 3)

    assert harmonic_mean == expected_hm


@pytest.mark.parametrize(
    "start_idx,end_idx,expected_hm",
    [(0, 3, 1501.535), (0, 6, 1501.07), (2, 5, 1500.915), (4, 7, 1500.295)],
)
def test__compute_hm(start_idx, end_idx, expected_hm):
    depth = np.arange(7) * 10
    speed = np.arange(1502, 1500, step=-0.31)

    svdf = pd.DataFrame({SP_DEPTH: depth, SP_SOUND_SPEED: speed})

    # Get partial of the data from test index
    partdf = svdf[start_idx:end_idx].copy()
    start_depth, end_depth = partdf.iloc[0][SP_DEPTH], partdf.iloc[-1][SP_DEPTH]
    result_hm = round(_compute_hm(svdf, start_depth, end_depth), 3)

    # Check for result to match expected
    assert result_hm == expected_hm

    # Result should be same as manual computation of weighted harmonic mean
    # https://en.wikipedia.org/wiki/Harmonic_mean#Weighted_harmonic_mean
    #
    # H = (w1+...+wn) / ((w1/x1)+...+(wn/xn))
    #
    # H is the resulting harmonic mean
    # w is the weight value, in this case, the depth differences
    # x is the input value, in this case, the sound speed
    w = partdf[SP_DEPTH].diff()
    x = partdf[SP_SOUND_SPEED]
    H = w.dropna().sum() / (w / x).dropna().sum()
    assert result_hm == round(H, 3)


# Add test for missing columns in dataframe
def test__compute_hm_missing_columns():
    svdf = pd.DataFrame({"random_column1": [1, 2, 3], "random_column2": [4, 5, 6]})
    start_depth = 0
    end_depth = 2
    with pytest.raises(ValueError):
        _compute_hm(svdf, start_depth, end_depth)


# Add test for empty dataframe
def test_sv_harmonic_mean_empty_abs_sv():
    # Create a DataFrame with empty 'abs_sv' column
    svdf = pd.DataFrame({"speed": [], "depth": []})
    start_depth = 0
    end_depth = 2
    with pytest.raises(ValueError):
        sv_harmonic_mean(svdf, start_depth, end_depth)


# Add test for empty dataframe
def test_sv_harmonic_mean_empty_dataframe():
    svdf = pd.DataFrame()
    start_depth = 0
    end_depth = 2
    with pytest.raises(ValueError):
        sv_harmonic_mean(svdf, start_depth, end_depth)
