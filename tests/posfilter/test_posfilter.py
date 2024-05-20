import numpy as np
import pandas as pd
import pytest

from gnatss import constants
from gnatss.loaders import load_gps_solutions
from gnatss.posfilter import rotation
from tests import TEST_DATA_FOLDER


@pytest.fixture
def kalman_solution_reference():
    pos_twtt_file = [(TEST_DATA_FOLDER / "2022" / "NCL1" / "POS_TWTT").absolute()]
    return load_gps_solutions(pos_twtt_file, from_legacy=True)


@pytest.fixture
def rotation_data(kalman_filtering_data, spline_interpolate_data, configuration):
    rotation_solutions = rotation(
        kalman_filtering_data,
        spline_interpolate_data,
        configuration.posfilter.atd_offsets,
        configuration.array_center,
    )
    return rotation_solutions


def test_spline_interpolate(
    novatel_data,
    novatel_std_data,
    travel_times_data,
    roll_pitch_heading_data,
    spline_interpolate_data,
):
    spline_interpolate_df = spline_interpolate_data[
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
                    merged_df[f"{col}_ref"] - merged_df[f"{col}_interpolated"] > rph_error_tol
                ]
            )
            == 0
        )


def test_kalman_filtering(
    kalman_filtering_data, travel_times_data, kalman_solution_reference
):
    pd.options.display.float_format = "{:.6f}".format
    # print(f"Travel Times\n{travel_times_data}")

    # kalman_solutions_df = kalman_filtering_data.iloc[:, [0,4,5,6,7,8,9,10,11,12,13,14,15]]
    kalman_solutions_df = kalman_filtering_data

    # print(f"Kalman Solution: {kalman_solutions_df.columns}\n{kalman_solutions_df}")

    kalman_ref_df = kalman_solution_reference
    # print(f"Kalman Reference\n{kalman_ref_df}")

    merged_df = kalman_ref_df.merge(
        kalman_solutions_df,
        how="inner",
        left_on=constants.GPS_TIME,
        right_on=constants.GPS_TIME,
        suffixes=("_ref", "_solutions"),
    )

    # print(f"Merged df\n{merged_df}")

    # df1 = merged_df[['x', 'ant_x', 'y', 'ant_y', 'z', 'ant_z']]
    print(f"Merged df\n{merged_df[['x', 'ant_x', 'y', 'ant_y', 'z', 'ant_z']]}")

    ref_cols = constants.GPS_GEOCENTRIC
    solutions_cols = constants.ANT_GPS_GEOCENTRIC

    rph_error_tol = 1e-2  # 1 cm error tolerance
    for ref_col, solutions_col in zip(ref_cols, solutions_cols):
        print(f"{ref_col} {solutions_col}")
        assert (
            len(
                merged_df[merged_df[ref_col] - merged_df[solutions_col] > rph_error_tol]
            )
            == 0
        )

    assert not merged_df.empty

    # assert np.allclose(merged_df.loc[:, "x"], merged_df.loc[:, "ant_x"])
    # assert np.allclose(merged_df.loc[:, "y"], merged_df.loc[:, "ant_y"])
    # assert np.allclose(merged_df.loc[:, "z"], merged_df.loc[:, "ant_z"])

    print(
        f"Merged df\n{merged_df[['xx', 'ant_cov_xx', 'yy', 'ant_cov_yy', 'zz', 'ant_cov_zz']]}"
    )

    sqrt_df = merged_df[["ant_cov_xx", "ant_cov_yy", "ant_cov_zz"]].pow(0.5).mul(100)
    print(f"sqrt df: \n{sqrt_df}")
    print(sqrt_df.min(axis=0))
    print(sqrt_df.max(axis=0))

    # assert kalman_filtering.empty()


def test_rotation(legacy_gps_solutions_data, rotation_data):
    pd.options.display.float_format = "{:.4f}".format

    rotations_solutions_df = rotation_data
    # print(f"rotations_soltions_df:{rotations_solutions_df.columns}\n\n{rotations_solutions_df}")

    rotations_ref_df = legacy_gps_solutions_data
    # print(f"rotations_ref_df:{rotations_ref_df.columns}\n\n{rotations_ref_df}")

    merged_df = rotations_ref_df.merge(
        rotations_solutions_df,
        how="inner",
        left_on=constants.GPS_TIME,
        right_on=constants.TIME_J2000,
        suffixes=("_ref", "_solutions"),
    )
    merged_df = merged_df[["x_ref", "ant_x", "xx", "ant_cov_xx"]]
    print(f"merged_df:{merged_df.columns}\n\n{merged_df}")

    ref_cols = [f"{col}_ref" for col in constants.GPS_GEOCENTRIC]
    solutions_cols = constants.ANT_GPS_GEOCENTRIC

    rph_error_tol = 1e-2  # 1 cm error tolerance
    for ref_col, solutions_col in zip(ref_cols, solutions_cols):
        print(f"{ref_col} {solutions_col}")
        assert (
            len(
                merged_df[merged_df[ref_col] - merged_df[solutions_col] > rph_error_tol]
            )
            == 0
        )
        # print(merged_df[[ref_col, solutions_col]])
        # assert np.allclose(merged_df.loc[:, ref_col], merged_df.loc[:, solutions_col], rtol=1e-4)
