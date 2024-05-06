import numpy as np
import pytest

from gnatss import constants
from gnatss.posfilter import spline_interpolate


@pytest.mark.parametrize(
    "full_result",
    [
        False,
    ],
)
def test_spline_interpolate(
    novatel_data,
    novatel_std_data,
    travel_times_data,
    roll_pitch_heading_data,
    full_result,
):
    spline_interpolate_df = spline_interpolate(
        novatel_data, novatel_std_data, travel_times_data, full_result=full_result
    )
    spline_interpolate_df = spline_interpolate_df[
        [constants.TT_TIME, *constants.RPH_LOCAL_TANGENTS]
    ]

    # Retrieve reference roll pitch heading data from files
    roll_pitch_heading_data = roll_pitch_heading_data[
        [constants.RPH_TIME, *constants.RPH_LOCAL_TANGENTS]
    ]

    merged_df = roll_pitch_heading_data.merge(
        spline_interpolate_df,
        how="inner",
        left_on=constants.RPH_TIME,
        right_on=constants.TT_TIME,
        suffixes=("_ref", "_interpolated"),
    )

    # Verify times in spline_interpolate_df match those in reference roll_pitch_heading_data
    assert len(merged_df) == len(spline_interpolate_df)
    assert np.allclose(merged_df.loc[:, "time"], spline_interpolate_df.loc[:, "time"])

    # Verify that interpolated roll pitch headings matches reference within 1e-5 units
    rph_error_tol = 1e-5
    for col in constants.RPH_LOCAL_TANGENTS:
        assert (
            len(
                merged_df[
                    merged_df[f"{col}_ref"] - merged_df[f"{col}_interpolated"]
                    > rph_error_tol
                ]
            )
            == 0
        )
