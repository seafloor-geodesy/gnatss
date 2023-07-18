import numba
import numpy as np
from numba.typed import List as NumbaList
from pymap3d import ecef2enu, ecef2geodetic

from .. import constants
from .utils import _prep_col_names

TRANSMIT_LOC_COLS = _prep_col_names(constants.GPS_GEOCENTRIC)
REPLY_LOC_COLS = _prep_col_names(constants.GPS_GEOCENTRIC, transmit=False)


@numba.njit(cache=True)
def _split_cov(cov_values):
    n = 3
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i] = cov_values[i * n : i * n + n]  # noqa
    return cov


def calc_lla_and_enu(all_observations, array_center):
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


def get_data_inputs(all_observations):
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
        _split_cov(row.to_numpy()) for idx, row in cov_vals_df.iterrows()
    ]

    # Merge all inputs
    for data in zip(
        transmit_xyz, reply_xyz_list, gps_covariance_matrices, observed_delay_list
    ):
        data_inputs.append(data)
    return data_inputs
