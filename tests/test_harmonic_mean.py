import numpy as np
import pandas as pd
import pytest

from gnatss.constants import SP_DEPTH, SP_SOUND_SPEED
from gnatss.harmonic_mean import _compute_hm, sv_harmonic_mean

from . import TEST_DATA_FOLDER


def sound_profile_data(delete_rows: bool = False) -> pd.DataFrame:
    """
    Returns dataframe parsed from sound_profile_hm.csv in test data folder.

    Parameters
    ----------
    delete_rows : bool
        If True, all the rows in the dataframe are deleted, while leaving columns intact.

    Returns
    -------
    pd.DataFrame
    """
    sv_file = f"{TEST_DATA_FOLDER}/sound_profile_hm.csv"
    df = pd.read_csv(sv_file)
    if delete_rows:
        df.drop(df.index, inplace=True)
    return df


@pytest.mark.parametrize(
    "end_depth,expected_hm,sound_profile_data,expected_exception_message",
    [
        (-1176.5866, 1481.542, sound_profile_data(), None),
        (-1146.5881, 1481.513, sound_profile_data(), None),
        (-1133.7305, 1481.5, sound_profile_data(), None),
        (
            -1133.7305,
            None,
            sound_profile_data(delete_rows=True),
            "Dataframe is empty! Please check your data inputs.",
        ),
    ],
)
def test_sv_harmonic_mean(
    end_depth, expected_hm, sound_profile_data, expected_exception_message
):
    svdf = sound_profile_data
    start_depth = -4

    if expected_exception_message:
        with pytest.raises(ValueError) as exc:
            _ = round(sv_harmonic_mean(svdf, start_depth, end_depth), 3)
            assert expected_exception_message in str(exc)

    else:
        harmonic_mean = round(sv_harmonic_mean(svdf, start_depth, end_depth), 3)
        assert harmonic_mean == expected_hm


@pytest.mark.parametrize(
    "start_idx,end_idx,expected_hm,df_cols",
    [
        (0, 3, 1501.535, (SP_DEPTH, SP_SOUND_SPEED)),
        (0, 6, 1501.07, (SP_DEPTH, SP_SOUND_SPEED)),
        (2, 5, 1500.915, (SP_DEPTH, SP_SOUND_SPEED)),
        (4, 7, 1500.295, (SP_DEPTH, SP_SOUND_SPEED)),
        (4, 7, 1500.295, (SP_DEPTH)),
        (4, 7, 1500.295, (SP_SOUND_SPEED)),
        (4, 7, 1500.295, tuple()),
    ],
)
def test__compute_hm(start_idx, end_idx, expected_hm, df_cols):
    depth = np.arange(7) * 10
    speed = np.arange(1502, 1500, step=-0.31)

    svdf = pd.DataFrame({SP_DEPTH: depth, SP_SOUND_SPEED: speed})

    # Get partial of the data from test index
    partdf = svdf[start_idx:end_idx].copy()
    start_depth, end_depth = partdf.iloc[0][SP_DEPTH], partdf.iloc[-1][SP_DEPTH]

    if df_cols == (SP_DEPTH, SP_SOUND_SPEED):
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

    else:
        if df_cols == (SP_DEPTH):
            svdf.drop([SP_SOUND_SPEED], axis=1, inplace=True)
        elif df_cols == (SP_SOUND_SPEED):
            svdf.drop([SP_DEPTH], axis=1, inplace=True)
        elif df_cols == tuple():
            svdf.drop([SP_DEPTH, SP_SOUND_SPEED], axis=1, inplace=True)

        with pytest.raises(ValueError) as exc:
            _ = round(_compute_hm(svdf, start_depth, end_depth), 3)
            assert "column must exist in the input dataframe!" in str(exc)
