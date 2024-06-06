from __future__ import annotations

import numpy as np
import pytest
from tests import TEST_DATA_FOLDER

from gnatss import constants
from gnatss.loaders import load_gps_solutions
from gnatss.posfilter import rotation


@pytest.fixture()
def kalman_solution_reference():
    pos_twtt_file = [(TEST_DATA_FOLDER / "2022" / "NCL1" / "POS_TWTT").absolute()]
    return load_gps_solutions(pos_twtt_file, from_legacy=True)


@pytest.fixture()
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


def test_kalman_filtering(kalman_filtering_data, travel_times_data, kalman_solution_reference):
    kalman_solutions_df = kalman_filtering_data
    kalman_ref_df = kalman_solution_reference

    merged_df = kalman_ref_df.merge(
        kalman_solutions_df,
        how="inner",
        left_on=constants.GPS_TIME,
        right_on=constants.GPS_TIME,
        suffixes=("_ref", "_solutions"),
    )

    assert not merged_df.empty

    # Reference GPS_GEOCENTRIC cols in reference should match solutions ANT_GPS_GEOCENTRIC cols within 1cm tolerance
    ref_cols = constants.GPS_GEOCENTRIC
    solutions_cols = constants.ANT_GPS_GEOCENTRIC
    col_error_tol = 1e-2
    for ref_col, solutions_col in zip(ref_cols, solutions_cols):
        assert (
            len(merged_df[(merged_df[ref_col] - merged_df[solutions_col]).abs() > col_error_tol])
            == 0
        )

    # Square root of ANT_GPS_COV_DIAG values should lie between 0.5cm to 5cm
    min_sqrt_value = 5e-3
    max_sqrt_value = 5e-2
    sqrt_df = merged_df[constants.ANT_GPS_COV_DIAG].pow(0.5)
    assert sqrt_df.min(axis=None) >= min_sqrt_value
    assert sqrt_df.max(axis=None) <= max_sqrt_value


def test_rotation(legacy_gps_solutions_data, rotation_data):
    rotations_solutions_df = rotation_data
    rotations_ref_df = legacy_gps_solutions_data

    merged_df = rotations_ref_df.merge(
        rotations_solutions_df,
        how="inner",
        left_on=constants.GPS_TIME,
        right_on=constants.TIME_J2000,
        suffixes=("_ref", "_solutions"),
    )

    # A maximum of 2.5% of rows can exceed the 1cm tolerance between the solutions and reference GPS_GEOCENTRIC cols
    ref_cols = [f"{col}_ref" for col in constants.GPS_GEOCENTRIC]
    solutions_cols = [f"{col}_solutions" for col in constants.GPS_GEOCENTRIC]
    rows_outside_threshold_percent = 2.5
    col_err_tol = 1e-2
    for ref_col, solutions_col in zip(ref_cols, solutions_cols):
        outlier_df = merged_df.loc[
            (merged_df[ref_col] - merged_df[solutions_col]).abs() > col_err_tol
        ]
        assert len(outlier_df) * 100.0 / len(merged_df) < rows_outside_threshold_percent
