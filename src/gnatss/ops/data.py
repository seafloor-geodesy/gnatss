from typing import List

import numba
import numpy as np
import pandas as pd
from nptyping import Float64, NDArray, Shape
from numba.typed import List as NumbaList
from pymap3d import ecef2enu, ecef2geodetic

from .. import constants
from ..configs.solver import ArrayCenter
from .utils import _prep_col_names

TRANSMIT_LOC_COLS = _prep_col_names(constants.GPS_GEOCENTRIC)
REPLY_LOC_COLS = _prep_col_names(constants.GPS_GEOCENTRIC, transmit=False)


@numba.njit(cache=True)
def _split_cov(
    cov_values: NDArray[Shape["9"], Float64]
) -> NDArray[Shape["3, 3"], Float64]:
    """
    Splits an array of covariance values of shape (9,)
    to a matrix array of shape (3, 3)

    Parameters
    ----------
    cov_values : (9,) ndarray
        Covariance array of shape (9,)
        in the order [xx, xy, xz,yx, yy, yx, yz, zx, zy, zz]

    Returns
    -------
    (3,3) ndarray
        The final covariance matrix of shape (3,3)
    """

    n = 3
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i] = cov_values[i * n : i * n + n]  # noqa
    return cov


def ecef_to_enu(
    df: pd.DataFrame,
    input_ecef_columns: List[str],
    output_enu_columns: List[str],
    array_center: ArrayCenter,
) -> pd.DataFrame:
    """
    Calculate ENU coordinates from input ECEF coordinates

    Parameters
    ----------
    df: pd.DataFrame
        The full dataset for computation
    input_ecef_columns: List[str]
        Columns in the df that contain ENU coordinates
    output_enu_columns: List[str]
        Columns that should be created in the df for ENU coordinates
    array_center : ArrayCenter
        An object containing the center of the array

    Returns
    -------
    pd.DataFrame
        Modified dataset with ECEF and ENU coordinates
    """
    enu = df[input_ecef_columns].apply(
        lambda row: ecef2enu(
            *row.values,
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
    )
    df = df.assign(**dict(zip(output_enu_columns, zip(*enu))))
    return df


def calc_lla_and_enu(
    all_observations: pd.DataFrame, array_center: ArrayCenter
) -> pd.DataFrame:
    """
    Calculates the LLA and ENU coordinates for all observations

    Parameters
    ----------
    all_observations : pd.DataFrame
        The full dataset for computation
    array_center : ArrayCenter
        An object containing the center of the array

    Returns
    -------
    pd.DataFrame
        Modified dataset with LLA and ENU coordinates
    """
    lla = all_observations[TRANSMIT_LOC_COLS].apply(
        lambda row: ecef2geodetic(*row.values), axis=1
    )
    enu = all_observations[TRANSMIT_LOC_COLS].apply(
        lambda row: ecef2enu(
            *row.values,
            lat0=array_center.lat,
            lon0=array_center.lon,
            h0=array_center.alt,
        ),
        axis=1,
    )
    all_observations = all_observations.assign(
        **dict(zip(_prep_col_names(constants.GPS_GEODETIC), zip(*lla)))
    )
    all_observations = all_observations.assign(
        **dict(zip(_prep_col_names(constants.GPS_LOCAL_TANGENT), zip(*enu)))
    )
    return all_observations


def get_data_inputs(all_observations: pd.DataFrame) -> NumbaList:
    """Extracts data inputs to perform solving algorithm

    Parameters
    ----------
    all_observations : pd.DataFrame
        The full dataset for computation

    Returns
    -------
    NumbaList
        A list of data inputs
    """
    # Set up special numba list so it can be passed
    # into numba functions for just in time compilation
    data_inputs = NumbaList()

    # Group obs by the transmit time
    grouped_obs = all_observations.groupby(constants.garpos.ST)

    # Get transmit xyz
    transmit_xyz = grouped_obs[TRANSMIT_LOC_COLS].first().to_numpy()

    # Get reply xyz
    reply_xyz_list = []
    grouped_obs[REPLY_LOC_COLS].apply(
        lambda group: reply_xyz_list.append(group.to_numpy())
    )

    # Get observed delays
    observed_delay_list = []
    grouped_obs[constants.garpos.TT].apply(
        lambda group: observed_delay_list.append(group.to_numpy())
    )

    # Get transmit cov matrices
    cov_vals_df = grouped_obs[_prep_col_names(constants.GPS_COV, True)].first()
    gps_covariance_matrices = [
        _split_cov(row.to_numpy()) for _, row in cov_vals_df.iterrows()
    ]

    # Merge all inputs
    for data in zip(
        transmit_xyz, reply_xyz_list, gps_covariance_matrices, observed_delay_list
    ):
        data_inputs.append(data)
    return data_inputs


def clean_tt(
    travel_times: pd.DataFrame,
    transponder_ids: List[str],
    travel_times_correction: float,
    transducer_delay_time: float,
) -> pd.DataFrame:
    """
    Clean travel times by doing the following steps:
    1. remove any travel times that have 0 reply time
    2. apply travel time correction and transducer delay time.

    Parameters
    ----------
    travel_times : pd.DataFrame
        The original travel times data
    transponder_ids : List[str]
        A list of the transponder ids that matches the order
        with travel_times data
    travel_times_correction : float
        Correction to times in travel times (secs.)
    transducer_delay_time : float
        Transducer Delay Time - delay at surface transducer (secs).

    Returns
    -------
    pd.DataFrame
        The cleaned travel times data
    """
    # Get cleaned travel times
    # This is anything that has 0 reply time
    cleaned_travel_times = travel_times.loc[
        travel_times[transponder_ids]
        .where(travel_times[transponder_ids] != 0)
        .dropna()
        .index
    ]

    # Apply travel time correction
    cleaned_travel_times.loc[:, constants.TT_TIME] = (
        cleaned_travel_times[constants.TT_TIME]
        + travel_times_correction
        + transducer_delay_time
    )

    # TODO: Store junk travel times? These are travel times with 0 values
    # _ = all_travel_times.loc[
    #     all_travel_times.where(all_travel_times[transponder_ids] == 0)
    #     .dropna(how="all")
    #     .index
    # ]

    return cleaned_travel_times


def filter_tt(travel_times: pd.DataFrame, cut_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter travel times data by removing the data that falls within
    the time range specified in the deletions file.

    Parameters
    ----------
    travel_times : pd.DataFrame
        The original travel times data
    cut_df : pd.DataFrame
        The deletions data to be removed

    Returns
    -------
    pd.DataFrame
        The filtered travel times data
    """

    if len(cut_df.index) > 0:
        # Only cut the data with deletions file if there are data
        cut_ids_all = []
        for _, cut in cut_df.iterrows():
            cut_ids = travel_times[
                (travel_times[constants.TT_TIME] >= cut.starttime)
                & (travel_times[constants.TT_TIME] <= cut.endtime)
            ].index.values
            cut_ids_all = cut_ids_all + cut_ids.tolist()
        cut_ids_all = list(set(cut_ids_all))
        return travel_times.loc[~travel_times.index.isin(cut_ids_all)]
    return travel_times


def preprocess_tt(travel_times: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess travel times data by creating a dataframe that
    contains the travel times and the reply times contiguously.

    Parameters
    ----------
    travel_times : pd.DataFrame
        The travel times data

    Returns
    -------
    pd.DataFrame
        The preprocessed travel times data
    """
    data = []
    for tx, tds in travel_times.set_index(constants.TT_TIME).iterrows():
        data.append((tx, constants.garpos.ST, 0, np.nan))
        for k, td in tds.to_dict().items():
            data.append((tx + td, k, td, tx))

    return pd.DataFrame(
        data,
        columns=[
            constants.TT_TIME,
            constants.garpos.MT,
            constants.garpos.TT,
            constants.garpos.ST,
        ],
    )
