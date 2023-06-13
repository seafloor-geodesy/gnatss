import numba
import numpy as np
from numba.typed import List as NumbaList
from pymap3d import ecef2enu, ecef2geodetic

from .. import constants
from .utils import _prep_col_names

TRANSMIT_LOC_COLS = _prep_col_names(constants.GPS_GEOCENTRIC)
REPLY_LOC_COLS = _prep_col_names(constants.GPS_GEOCENTRIC, transmit=False)


@numba.njit
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


def get_data_inputs(all_observations, config):
    data_inputs = NumbaList()
    for _, group in all_observations.groupby(constants.garpos.ST):
        transmit_xyz = group[TRANSMIT_LOC_COLS].values[0]
        reply_xyz = group[REPLY_LOC_COLS].values

        # Get observed delays
        observed_delays = group[constants.garpos.TT].values

        # Get transmit cov matrix
        cov_values = group[_prep_col_names(constants.GPS_COV, True)].values[0]
        gps_covariance_matrix = _split_cov(cov_values)

        # Get travel times variance
        travel_times_variance = config.solver.travel_times_variance
        data_inputs.append(
            (
                transmit_xyz,
                reply_xyz,
                gps_covariance_matrix,
                observed_delays,
                travel_times_variance,
            )
        )
    return data_inputs
