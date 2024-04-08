from typing import List

import pandas as pd
from nptyping import Float, NDArray, Shape
from pymap3d import ecef2enu
from scipy.spatial.transform import Rotation

from .. import constants
from ..configs.solver import ArrayCenter


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
    df :  DataFrame
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
