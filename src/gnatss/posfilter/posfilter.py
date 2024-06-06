# ruff: noqa: PD901
from __future__ import annotations

import numpy as np
import pandas as pd
import pymap3d
from nptyping import Float, NDArray, Shape
from scipy.spatial.transform import Rotation

from .. import constants
from ..configs.posfilter import AtdOffset
from ..configs.solver import ArrayCenter
from ..ops.kalman import run_filter_simulation


def spline_interpolate(
    inspvaa_df: pd.DataFrame,
    insstdeva_df: pd.DataFrame,
    twtt_df: pd.DataFrame,
    full_result: bool = False,
) -> pd.DataFrame:
    """
    Interpolate the INS RPH data and the covariance data to the travel times data

    Parameters
    ----------
    inspvaa_df :  DataFrame
        Pandas Dataframe containing Antenna enu directions Novatel Level-1 data
    insstdeva_df :  DataFrame
        Pandas Dataframe containing Antenna enu directions std deviation Novatel Level-1 data
    twtt_df : pd.DataFrame
        The travel times data
    full_result : bool
        TODO: Add description

    Returns
    -------
    cov_rph_twtt : pd.DataFrame
        The interpolated data
    """
    ins_rph = inspvaa_df[[constants.RPH_TIME, *constants.RPH_LOCAL_TANGENTS]]
    cov_rph = insstdeva_df[[constants.RPH_TIME, *constants.PLATFORM_COV_RPH_DIAG]]
    # Merge the two dataframes of ins_rph and cov_rph
    # since they're at the same sampling rate
    merged_rph = ins_rph.merge(cov_rph, on=constants.RPH_TIME)

    # Merge the travel times data with the merged_rph data
    # with an 'outer' join, this will result in missing values
    # at points of travel times data that doesn't have corresponding
    # INS RPH data
    # TODO: Confirm why we are using TT_TIME, GPS_TIME, and RPH_TIME for merge
    initial_df = twtt_df[[constants.TT_TIME]].merge(
        merged_rph,
        left_on=constants.GPS_TIME,
        right_on=constants.RPH_TIME,
        how="outer",
    )

    # Interpolate the missing values using cubic spline interpolation
    result_df = (
        initial_df.interpolate(method="spline", order=3, s=0).dropna().reset_index(drop=True)
    )

    for col in constants.PLATFORM_COV_RPH:
        if col not in constants.PLATFORM_COV_RPH_DIAG:
            result_df[col] = 0.0

    # TODO: Confirm: return values df has different columns depending on full_results
    if full_result:
        return result_df

    return twtt_df.merge(result_df, on=constants.TT_TIME, how="left")


def rotation(
    pos_twtt: pd.DataFrame,
    cov_rph_twtt: pd.DataFrame,
    atd_offsets: AtdOffset | NDArray[Shape[3], Float],
    array_center: ArrayCenter,
) -> pd.DataFrame:
    """
    Calculate GNSS antenna eastward, northward, and upward position
    columns,and add to input dataframe.

    Parameters
    ----------
    pos_twtt :  pd.DataFrame
        Pandas Dataframe containing positioning data as result from
        the kalman filtering process
    cov_rph_twtt : pd.DataFrame
        Pandas Dataframe containing INS RPH data, including the covariance data
        from the spline interpolation process
    atd_offsets : NDArray[Shape["3"], Float]
        Forward, Rightward, and Downward antenna transducer offset values
    array_center : ArrayCenter
        Array center base model containing Latitude, Longitude and Altitude

    Returns
    -------
    pos_freed_trans_twtt : pd.DataFrame
        Modified positioning data that includes rotation
    """
    # Merge the pos_twtt and cov_rph_twtt dataframes
    df = pos_twtt.merge(
        cov_rph_twtt,
        on=constants.TIME_J2000,
        how="left",
        suffixes=("", "_remove"),
    )
    # remove the duplicate columns
    df = df.drop([i for i in df.columns if "remove" in i], axis=1)

    # Extract atd_offsets values
    if isinstance(atd_offsets, AtdOffset):
        atd_offsets = np.array([atd_offsets.forward, atd_offsets.rightward, atd_offsets.downward])

    # d_enu_columns and td_enu_columns are temporary columns used to calculate antenna positions
    d_enu_columns = ["d_e", "d_n", "d_u"]
    td_enu_columns = ["td_e", "td_n", "td_u"]

    # Compute transducer offset from the antenna, and add to d_enu_columns columns
    r = Rotation.from_euler("xyz", df[constants.RPH_LOCAL_TANGENTS], degrees=True)

    # Compute the final offsets
    offsets = r.as_matrix() @ atd_offsets
    df[d_enu_columns[0]] = offsets[:, 1]
    df[d_enu_columns[1]] = offsets[:, 0]
    df[d_enu_columns[2]] = -offsets[:, 2]

    # Calculate enu values from ecef values, and add to td_enu_columns columns
    df[td_enu_columns] = df[constants.ANT_GPS_GEOCENTRIC].apply(
        lambda row: pymap3d.ecef2enu(
            *row.to_numpy(),
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
        result_type="expand",
    )

    transducer_columns = constants.GPS_GEOCENTRIC
    # antenna_enu is the sum of corresponding td_enu_columns and d_enu_columns values
    for trans_enu, td_enu, d_enu in zip(
        constants.GPS_LOCAL_TANGENT, td_enu_columns, d_enu_columns, strict=False
    ):
        df[trans_enu] = df.loc[:, [td_enu, d_enu]].sum(axis=1)

    # convert to ecef coordinates
    df[transducer_columns] = df[constants.GPS_LOCAL_TANGENT].apply(
        lambda row: pymap3d.enu2ecef(
            *row.to_numpy(),
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
        result_type="expand",
    )

    # Drop temporary columns
    return df.drop(columns=[*d_enu_columns, *td_enu_columns, *constants.GPS_LOCAL_TANGENT])


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
        Pandas Dataframe containing Antenna enu directions Novatel Level-1 data
    insstdeva_df :  pd.DataFrame
        Pandas Dataframe containing Antenna enu directions std deviation Novatel Level-1 data
    gps_df :  pd.DataFrame
        Pandas Dataframe containing GPS solutions Novatel Level-1 data
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
            *constants.GPS_LOCAL_TANGENT,
        ]
    ]
    insstdeva_df = insstdeva_df.rename(
        columns={
            constants.TIME_J2000: constants.GPS_TIME,
        },
        errors="raise",
    )
    insstdeva_df = insstdeva_df[
        [constants.GPS_TIME] + [f"{d}_sig" for d in constants.GPS_LOCAL_TANGENT]
    ]

    insstdeva_df["v_sden"] = 0.0  # TODO Should I create a constant for these columns?
    insstdeva_df["v_sdeu"] = 0.0
    insstdeva_df["v_sdnu"] = 0.0

    # GPS Position correlation coefficients
    gps_df["rho_xy"] = 0.0
    gps_df["rho_xz"] = 0.0
    gps_df["rho_yz"] = 0.0

    merged_df = inspvaa_df.merge(twtt_df[[constants.GPS_TIME]], on=constants.GPS_TIME, how="outer")
    merged_df = merged_df.merge(gps_df, on=constants.GPS_TIME, how="left")
    merged_df = merged_df.merge(insstdeva_df, on=constants.GPS_TIME, how="left")
    merged_df = merged_df.sort_values(constants.GPS_TIME).reset_index(drop=True)

    first_pos = merged_df[~merged_df[constants.ANT_GPS_GEOCENTRIC[0]].isna()].iloc[0].name
    merged_df = merged_df.loc[first_pos:].reset_index(drop=True)

    merged_np_array = merged_df.to_numpy()
    # x: state matrix
    # P: covariance matrix of the predicted state
    # K: Kalman gain
    # Pp: predicted covariance from the RTS smoother
    x, P, _, _ = run_filter_simulation(merged_np_array, start_dt, gnss_pos_psd, vel_psd, cov_err)

    # Positions covariance
    ant_cov = P[:, :3, :3]
    ant_cov_df = pd.DataFrame(ant_cov.reshape(ant_cov.shape[0], -1), columns=constants.ANT_GPS_COV)
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
    smoothed_results = smoothed_results.merge(ant_cov_df, on=constants.GPS_TIME, how="left")

    if full_result:
        return smoothed_results

    return twtt_df.merge(smoothed_results, how="left")
