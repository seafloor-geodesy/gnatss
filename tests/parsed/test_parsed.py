from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame
from pandas.api.types import is_float_dtype, is_integer_dtype, is_object_dtype, is_string_dtype
from tests import TEST_DATA_FOLDER

from gnatss import constants
from gnatss.loaders import load_gps_solutions
from gnatss.parsed import get_parsed_ant_positions
from gnatss.posfilter import rotation


def test_parsed_cov_rph_twtt(parsed_cov_rph_twtt, raw_data_parsed):

    expected_columns = [
        constants.DATA_SPEC.transponder_id,
        constants.RPH_TIME,
        constants.DATA_SPEC.tx_time,
        constants.DATA_SPEC.travel_time,
        *constants.RPH_LOCAL_TANGENTS,
        *constants.PLATFORM_COV_RPH
    ]

    # Test that data is in the expected format
    assert isinstance(parsed_cov_rph_twtt, DataFrame)
    assert set(expected_columns) == set(parsed_cov_rph_twtt.columns.values.tolist())
    assert is_string_dtype(parsed_cov_rph_twtt[constants.DATA_SPEC.transponder_id])
    assert all(
        is_float_dtype(parsed_cov_rph_twtt[column]) for column in expected_columns[1:]
    )

    # Test that data is not lost in function
    cov_rph_df = parsed_cov_rph_twtt[
        [constants.RPH_TIME, *constants.RPH_LOCAL_TANGENTS]
    ]

    # Retrieve reference roll pitch heading data from files
    roll_pitch_heading_data = raw_data_parsed[
        [constants.RPH_TIME, *constants.RPH_LOCAL_TANGENTS]
    ]

    merged_df = roll_pitch_heading_data.merge(
        cov_rph_df,
        how="inner",
        left_on=constants.RPH_TIME,
        right_on=constants.RPH_TIME,
        suffixes=("_ref", "_interpolated"),
    )

    # Verify times in spline_interpolate_df match those in reference roll_pitch_heading_data
    assert len(merged_df) == len(cov_rph_df)
    assert np.allclose(merged_df.loc[:, "time"], cov_rph_df.loc[:, "time"])

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


def test_parsed_pos_twtt(parsed_pos_twtt,gps_positions_parsed):

    expected_columns = [
        constants.DATA_SPEC.transponder_id,
        constants.GPS_TIME,
        constants.DATA_SPEC.tx_time,
        constants.DATA_SPEC.travel_time,
        *constants.ANT_GPS_GEOCENTRIC,
        *constants.ANT_GPS_GEOCENTRIC_STD,
        *constants.ANT_GPS_COV
    ]

    # Test that data is in the expected format
    assert isinstance(parsed_pos_twtt, DataFrame)
    assert set(expected_columns) == set(parsed_pos_twtt.columns.values.tolist())
    assert is_string_dtype(parsed_pos_twtt[constants.DATA_SPEC.transponder_id])
    assert all(
        is_float_dtype(parsed_pos_twtt[column]) for column in expected_columns[1:]
    )

    # Test that uncertainties are properly imported

    pos_twtt_df = parsed_pos_twtt[
        [constants.GPS_TIME, *constants.ANT_GPS_GEOCENTRIC_STD]
    ]

    # Retrieve position uncertainties from files
    gps_data_df = gps_positions_parsed[
        [constants.GPS_TIME, *constants.ANT_GPS_GEOCENTRIC_STD]
    ]
    

    merged_df = gps_data_df.merge(
        pos_twtt_df,
        how="inner",
        left_on=constants.GPS_TIME,
        right_on=constants.GPS_TIME,
        suffixes=("_ref", "_interpolated"),
    )

    # Verify that standard deviations match reference within 1e-5 units
    gps_error_tol = 1e-5
    for col in constants.ANT_GPS_GEOCENTRIC_STD:
        assert (
            len(
                merged_df[
                    merged_df[f"{col}_ref"] - merged_df[f"{col}_interpolated"] > gps_error_tol
                ]
            )
            == 0
        )


def test_parsed_pos_twtt_no_gps(raw_data_parsed):

    # Calculate pos_twtt with no input gps positions
    pos_twtt_df = get_parsed_ant_positions(raw_data_parsed)

    expected_columns = [
        constants.DATA_SPEC.transponder_id,
        constants.GPS_TIME,
        constants.DATA_SPEC.tx_time,
        constants.DATA_SPEC.travel_time,
        *constants.ANT_GPS_GEOCENTRIC,
        *constants.ANT_GPS_GEOCENTRIC_STD,
        *constants.ANT_GPS_COV
    ]

    # Test that data is in the expected format
    assert isinstance(pos_twtt_df, DataFrame)
    assert set(expected_columns) == set(pos_twtt_df.columns.values.tolist())
    assert is_string_dtype(pos_twtt_df[constants.DATA_SPEC.transponder_id])
    assert all(
        is_float_dtype(pos_twtt_df[column]) for column in expected_columns[1:]
    )

    # Test that data is not lost in function

    # Retrieve INS positioning data from files
    ins_data_df = raw_data_parsed[
        [constants.GPS_TIME, "ant_x1", "ant_y1", "ant_z1",]
    ]
    ins_data_df = ins_data_df.rename(columns={"ant_x1":"ant_x", "ant_y1":"ant_y", "ant_z1":"ant_z"})
        

    merged_df = ins_data_df.merge(
        pos_twtt_df,
        how="inner",
        left_on=constants.GPS_TIME,
        right_on=constants.GPS_TIME,
        suffixes=("_ref", "_interpolated"),
    )

    # Verify times in spline_interpolate_df match those in reference roll_pitch_heading_data
    assert len(merged_df) == len(pos_twtt_df)
    assert np.allclose(merged_df.loc[:, "time"], pos_twtt_df.loc[:, "time"])

    # Verify that interpolated roll pitch headings matches reference within 1e-5 units
    gps_error_tol = 1e-5
    for col in constants.ANT_GPS_GEOCENTRIC:
        assert (
            len(
                merged_df[
                    merged_df[f"{col}_ref"] - merged_df[f"{col}_interpolated"] > gps_error_tol
                ]
            )
            == 0
        )
