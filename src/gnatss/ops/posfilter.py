from typing import List

import numpy as np
import pandas as pd
from nptyping import Float, NDArray, Shape
from pymap3d import ecef2enu
from scipy.spatial.transform import Rotation

from .. import constants
from ..configs.solver import ArrayCenter
from .kalman import run_filter_simulation


def spline_interpolate(travel_times, ins_rph, cov_rph):
    """
    Interpolate the INS RPH data and the covariance data to the travel times data

    Parameters
    ----------
    travel_times : pd.DataFrame
        The travel times data
    ins_rph : pd.DataFrame
        The INS RPH data
    cov_rph : pd.DataFrame
        The covariance data

    Returns
    -------
    pd.DataFrame
        The interpolated data
    """
    # Merge the two dataframes of ins_rph and cov_rph
    # since they're at the same sampling rate
    merged_rph = pd.merge(ins_rph, cov_rph, on="dts")

    # Merge the travel times data with the merged_rph data
    # with an 'outer' join, this will result in missing values
    # at points of travel times data that doesn't have corresponding
    # INS RPH data
    initial_df = pd.merge(
        travel_times[[constants.TT_TIME]],
        merged_rph,
        left_on="time",
        right_on="dts",
        how="outer",
    )

    # Interpolate the missing values using cubic spline interpolation
    result_df = (
        initial_df.interpolate(method="spline", order=3, s=0)
        .drop("dts", axis=1)
        .dropna()
        .reset_index(drop=True)
    )

    return result_df


def rotation(
    df: pd.DataFrame,
    atd_offsets: NDArray[Shape["3"], Float],
    array_center: ArrayCenter,
    input_rph_columns: List[str],
    input_ecef_columns: List[str],
    output_antenna_enu_columns: List[str],
) -> pd.DataFrame:
    """
    Calculate GNSS antenna eastward, northward, and upward position
    columns,and add to input dataframe.

    Parameters
    ----------
    df :  pd.DataFrame
        Pandas Dataframe to add antenna position columns to
    atd_offsets : NDArray[Shape["3"], Float]
        Forward, Rightward, and Downward antenna transducer offset values
    array_center : ArrayCenter
        Array center base model containing Latitude, Longitude and Altitude
    input_rph_columns : List[str]
        List containing Roll, Pitch, and Heading column names in input dataframe
    input_ecef_columns : List[str]
        List containing Geocentric x, y, and z column names in input dataframe
    output_antenna_enu_columns : List[str]
        List containing Antenna position Eastward, Northward, and Upward
        direction column names that are to be added to the dataframe

    Returns
    -------
    pd.DataFrame
        Modified Pandas Dataframe containing 3 new antenna position
        (eastward, northward, and upward) columns.
    """
    # d_enu_columns and td_enu_columns are temporary columns used to calculate antenna positions
    d_enu_columns = ["d_e", "d_n", "d_u"]
    td_enu_columns = ["td_e", "td_n", "td_u"]

    # Compute transducer offset from the antenna, and add to d_enu_columns columns
    r = Rotation.from_euler("xyz", df[input_rph_columns], degrees=True)
    offsets = r.as_matrix() @ atd_offsets
    df[d_enu_columns[0]] = offsets[:, 1]
    df[d_enu_columns[1]] = offsets[:, 0]
    df[d_enu_columns[2]] = -offsets[:, 2]

    # Calculate enu values from ecef values, and add to td_enu_columns columns
    enu = df[input_ecef_columns].apply(
        lambda row: ecef2enu(
            *row.values,
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
    )
    df = df.assign(**dict(zip(td_enu_columns, zip(*enu))))

    # antenna_enu is the sum of corresponding td_enu_columns and d_enu_columns values
    for antenna_enu, td_enu, d_enu in zip(
        output_antenna_enu_columns, td_enu_columns, d_enu_columns
    ):
        df[antenna_enu] = df.loc[:, [td_enu, d_enu]].sum(axis=1)

    # Drop temporary d_enu_columns and td_enu_columns columns
    df.drop(columns=[*d_enu_columns, *td_enu_columns], inplace=True)

    return df


def kalman_filtering(
    inspvaa_df: pd.DataFrame,
    insstdeva_df: pd.DataFrame,
    gps_df: pd.DataFrame,
    twtt_df: pd.DataFrame,
    gnss_pos_psd=constants.gnss_pos_psd,
    vel_psd=constants.vel_psd,
    cov_err=constants.cov_err,
    start_dt=constants.start_dt,
    full_result: bool = False,
) -> pd.DataFrame:
    """
    Performs Kalman filtering of the GPS_GEOCENTRIC and GPS_COV_DIAG fields

    Parameters
    ----------
    inspvaa_df :  pd.DataFrame
        Pandas Dataframe containing Antenna enu directions Novatel L1 data
    insstdeva_df :  pd.DataFrame
        Pandas Dataframe containing Antenna enu directions std deviation Novatel L1 data
    gps_df :  pd.DataFrame
        Pandas Dataframe containing GPS solutions Novatel L1 data
    twtt_df :  pd.DataFrame
        Pandas Dataframe containing two way travel time data
    gnss_pos_psd : float
        GNSS Position Estimation Noise for creating Q Matrix
    vel_psd : float
        Velocity Estimation Noise for creating Q Matrix
    cov_err : float
        Initial state error covariance values for creating P Matrix
    start_dt : float
        Initial time step for creating F Matrix
    full_result : bool
        If True, returns the full result of the Kalman filter simulation
        rather than just data from the two way travel time timestamps

    Returns
    -------
    pos_twtt : pd.DataFrame
        Pandas Dataframe containing the resulting
        antenna positions, covariance, and standard deviation
    """
    # Instrument velocity data
    inspvaa_df = inspvaa_df.rename(
        columns={
            # TODO For merging convenience (So extra cols dont pop up during merge)
            constants.TIME_J2000: constants.GPS_TIME,
        },
        errors="raise",
    )
    inspvaa_df = inspvaa_df[
        [
            constants.GPS_TIME,
            *constants.ANTENNA_DIRECTIONS,
        ]
    ]
    insstdeva_df = insstdeva_df.rename(
        columns={
            constants.TIME_J2000: constants.GPS_TIME,
        },
        errors="raise",
    )
    insstdeva_df = insstdeva_df[
        [
            constants.GPS_TIME,
            f"{constants.ANTENNA_EASTWARD} std",
            f"{constants.ANTENNA_NORTHWARD} std",
            f"{constants.ANTENNA_UPWARD} std",
        ]
    ]

    insstdeva_df["v_sden"] = 0.0  # TODO Should I create a constant for these columns?
    insstdeva_df["v_sdeu"] = 0.0
    insstdeva_df["v_sdnu"] = 0.0

    # GPS Position correlation coefficients
    gps_df["rho_xy"] = 0.0
    gps_df["rho_xz"] = 0.0
    gps_df["rho_yz"] = 0.0

    merged_df = inspvaa_df.merge(
        twtt_df[[constants.GPS_TIME]], on=constants.GPS_TIME, how="outer"
    )
    merged_df = merged_df.merge(gps_df, on=constants.GPS_TIME, how="left")
    merged_df = merged_df.merge(insstdeva_df, on=constants.GPS_TIME, how="left")
    merged_df = merged_df.sort_values(constants.GPS_TIME).reset_index(drop=True)

    first_pos = merged_df[~merged_df.x.isnull()].iloc[0].name
    merged_df = merged_df.loc[first_pos:].reset_index(drop=True)

    merged_np_array = merged_df.to_numpy()

    # x: state matrix
    # P: covariance matrix of the predicted state
    # K: Kalman gain
    # Pp: predicted covariance from the RTS smoother
    x, P, _, _ = run_filter_simulation(
        merged_np_array, start_dt, gnss_pos_psd, vel_psd, cov_err
    )

    # Positions covariance
    ant_cov = P[:, :3, :3]
    ant_cov_df = pd.DataFrame(
        ant_cov.reshape(ant_cov.shape[0], -1), columns=constants.ANT_GPS_COV
    )
    ant_cov_df[[*constants.ANT_GPS_GEOCENTRIC_STD]] = ant_cov_df[
        [*constants.ANT_GPS_COV_DIAG]
    ].apply(np.sqrt)
    ant_cov_df[constants.GPS_TIME] = merged_df[constants.GPS_TIME]

    # Smoothed positions
    smoothed_results = pd.DataFrame(
        x.reshape(x.shape[0], -1)[:, :3],
        columns=constants.ANT_GPS_GEOCENTRIC,
    )
    smoothed_results[constants.GPS_TIME] = merged_df[constants.GPS_TIME]
    smoothed_results = smoothed_results.merge(
        ant_cov_df, on=constants.GPS_TIME, how="left"
    )

    if full_result:
        return smoothed_results

    return pd.merge(twtt_df, smoothed_results, how="left")
