from typing import List

import numpy as np
from nptyping import Float, NDArray, Shape
from pandas import DataFrame
from pymap3d import ecef2enu
from scipy.spatial.transform import Rotation

from gnatss import constants
from gnatss.configs.solver import ArrayCenter
from gnatss.loaders import load_gps_solutions
from gnatss.ops.kalman import run_filter_simulation


def rotation(
    df: DataFrame,
    atd_offsets: NDArray[Shape["3"], Float],
    array_center: ArrayCenter,
    input_rph_columns: List[str],
    input_ecef_columns: List[str],
    output_antenna_enu_columns: List[str],
) -> DataFrame:
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
    DataFrame
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
    inspvaa_df: DataFrame,
    insstdeva_df: DataFrame,
    gps_df: DataFrame,
):
    # gps_df = gps_df.rename(
    #     columns={
    #         "time": "dts",
    #     },
    #     errors="raise",
    # )

    gps_df = gps_df[
        [constants.GPS_TIME, *constants.GPS_GEOCENTRIC, "sdx", "sdy", "sdz"]
    ]

    inspvaa_df = inspvaa_df.rename(
        columns={
            constants.TIME_J2000: constants.GPS_TIME,  # For merging convenience (So extra cols dont pop up during merge)
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
            # f"{constants.ANTENNA_EASTWARD} std": "v_sde",
            # f"{constants.ANTENNA_NORTHWARD} std": "v_sdn",
            # f"{constants.ANTENNA_UPWARD} std": "v_sdu",
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

    insstdeva_df["v_sden"] = 0.0
    insstdeva_df["v_sdeu"] = 0.0
    insstdeva_df["v_sdnu"] = 0.0

    gps_df["sdxy"] = np.sqrt(gps_df.sdx * gps_df.sdy)
    gps_df["sdxz"] = np.sqrt(gps_df.sdx * gps_df.sdz)
    gps_df["sdyz"] = np.sqrt(gps_df.sdy * gps_df.sdz)

    merged_df = inspvaa_df.merge(gps_df, on=constants.GPS_TIME, how="left")
    merged_df = merged_df.merge(insstdeva_df, on=constants.GPS_TIME, how="left")
    merged_df = merged_df.sort_values(constants.GPS_TIME).reset_index(drop=True)

    first_pos = merged_df[~merged_df.x.isnull()].iloc[0].name
    merged_df = merged_df.loc[first_pos:].reset_index(drop=True)

    records = merged_df.to_numpy()
    x, P, K, Pp = run_filter_simulation(records)

    smoothed_results = DataFrame(
        x.reshape(x.shape[0], -1),
        columns=[*constants.GPS_GEOCENTRIC, "sdx", "sdy", "sdz"],
    )
    smoothed_results[constants.GPS_TIME] = merged_df[constants.GPS_TIME]
    # smoothed_results = smoothed_results[["dts", "x", "y", "z"]]
    # How are the GPS_COV values calculated???

    return smoothed_results
