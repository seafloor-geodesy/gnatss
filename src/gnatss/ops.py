from typing import Tuple

import numba
import numpy as np
import pandas as pd
import scipy
from nptyping import Float64, NDArray, Shape

from .configs.solver import ArrayCenter
from .constants import (
    GPS_COV_DIAG,
    GPS_GEOCENTRIC,
    GPS_GEODETIC,
    GPS_LOCAL_TANGENT,
    GPS_TIME,
)
from .utilities.geo import geocentric2enu, geocentric2geodetic


def find_gps_record(
    gps_solutions: pd.DataFrame, travel_time: pd.Timestamp
) -> pd.Series:
    """
    Finds matching GPS record based on travel time
    """
    match = gps_solutions.iloc[
        gps_solutions[GPS_TIME]
        .apply(lambda row: (row - travel_time))
        .abs()
        .argsort()[0],
        :,
    ]

    return match


def calc_std_and_verify(
    gps_series: pd.Series, std_dev: bool = True, sigma_limit: float = 0.05
) -> float:
    """
    Calculate the 3d standard deviation and verify the value based on limit

    Parameters
    ----------
    gps_series : pd.Series
        The data input to check as a pandas series.
        The following keys are expected: 'xx', 'yy', 'zz'
    std_dev : bool
        Flag to indicate if the inputs are standard deviation or variance
    sigma_limit : float
        The allowable sigma limit to check against

    Returns
    -------
    float
        The calculated sigma 3d

    Raises
    ------
    ValueError
        If 3D Standard Deviation exceeds the GPS Sigma limit
    """
    # Checks for GPS Covariance Diagonal values
    if not all(item in gps_series for item in GPS_COV_DIAG):
        raise KeyError(f"Not all values for {','.join(GPS_COV_DIAG)} exists.")

    # Compute the 3d std (sum variances of GPS components and take sqrt)
    sig_3d = np.sqrt(np.sum(gps_series[GPS_COV_DIAG] ** (2 if std_dev else 1)))

    # Verify sigma value, throw error if greater than gps sigma limit
    if sig_3d > sigma_limit:
        raise ValueError(
            f"3D Standard Deviation of {sig_3d} exceeds GPS Sigma Limit of {sigma_limit}!"
        )

    return sig_3d


def compute_enu_series(input_series: pd.Series, array_center: ArrayCenter) -> pd.Series:
    array_center_coords = [array_center.lon, array_center.lat, array_center.alt]

    location_series = input_series.copy()

    geodetic_coords = geocentric2geodetic(*location_series[GPS_GEOCENTRIC].values)
    enu_coords = geocentric2enu(
        *location_series[GPS_GEOCENTRIC].values, *array_center_coords
    ).flatten()

    # Set geodetic lon,lat,alt to the series
    for idx, v in enumerate(geodetic_coords):
        location_series[GPS_GEODETIC[idx]] = v

    # Set local tangent e,n,u to the series
    for idx, v in enumerate(enu_coords):
        location_series[GPS_LOCAL_TANGENT[idx]] = v

    return location_series


def calc_uv(input_vector: np.ndarray) -> np.ndarray:
    """Calculate unit vector"""

    if len(input_vector.shape) > 1:
        raise ValueError("Unit vector calculation must be 1-D array!")

    vector_norm = np.linalg.norm(input_vector)

    if vector_norm == 0:
        return np.array([2.0, 0.0, 0.0])

    return input_vector / vector_norm


@numba.njit
def calc_twtt_model(
    transmit_vectors: NDArray[Shape["3, *"], Float64],
    reply_vectors: NDArray[Shape["3, *"], Float64],
    transponders_mean_sv: NDArray[Shape["*"], Float64],
) -> NDArray[Shape["*"], Float64]:
    """
    Calculate the Modeled TWTT (Two way travel time) in seconds


    Parameters
    ----------
    transmit_vector : (3,N) ndarray
        The transmit array of vectors
    reply_vector : (3,N) ndarray
        The reply array of vectors
    transponders_mean_sv : (N,) ndarray
        The transponders mean sound speed

    Returns
    -------
    (N,) ndarray
        The modeled two way travel times in seconds

    """
    # Calculate distances in meters
    transmit_distance = np.array(
        [np.linalg.norm(vector) for vector in transmit_vectors]
    )
    reply_distance = np.array([np.linalg.norm(vector) for vector in reply_vectors])

    return (transmit_distance + reply_distance) / transponders_mean_sv


def calc_tt_residual(
    delays, transponder_delays, twtt_model
) -> NDArray[Shape["*"], Float64]:
    """
    Calculate the travel time residual in seconds

    Parameters
    ----------
    delays : (N,) ndarray
        The measured travel time delays from data (sec)
    transponder_delays : (N,) ndarray
        The set transponder delays as defined in configuration file (sec)
    twtt_model : (N,) ndarray
        The modeled two way travel times (sec)

    Returns
    -------
    (N,) ndarray
        The travel time residual
    """

    # dA = Ameas - Amod
    return (delays - transponder_delays) - twtt_model


def calc_partials(
    transmit_vectors: NDArray[Shape["3, *"], Float64],
    reply_vectors: NDArray[Shape["3, *"], Float64],
    transponders_mean_sv: NDArray[Shape["*"], Float64],
    delays: NDArray[Shape["*"], Float64],
    num_transponders: int,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Calculate the partial derivative matrices
    """

    transmit_uv = np.array([calc_uv(v) for v in transmit_vectors])
    reply_uv = np.array([calc_uv(v) for v in reply_vectors])

    # Calculate the partial derivatives
    partial_derivatives = (transmit_uv + reply_uv) / transponders_mean_sv
    A_partials = np.zeros(shape=(num_transponders, num_transponders * 3))
    B_cov = np.zeros(shape=(num_transponders, num_transponders * 3))

    # computing non-differenced, direct ranging
    # setup A partials and B covariance
    for i in range(num_transponders):
        if delays[i] > 0:
            start_idx = i * num_transponders
            end_idx = start_idx + 3  # Add 3 for x,y,z components
            A_partials[i, start_idx:end_idx] = partial_derivatives[i]
            B_cov[i, i] = 1.0

    return A_partials, B_cov, transmit_uv, reply_uv


def calc_weight_matrix(
    transmit_uv,
    gps_covariance_matrix,
    transponders_mean_sv,
    b_cov,
    travel_times_variance,
):
    """Calculate the weight matrix"""
    # Calculate covariance matrix for partlp vectors (COVF) Units m^2
    covariance_matrix = np.abs((transmit_uv @ gps_covariance_matrix @ transmit_uv.T))

    # Weighting factor equal to 2/sv_mean
    weight_factor = 2.0 / (transponders_mean_sv**2)

    # Convert to covariance of acoustic delays Units seconds^2
    covariance_matrix = weight_factor * covariance_matrix

    # Add travel times variance
    np.fill_diagonal(
        covariance_matrix, covariance_matrix.diagonal() + travel_times_variance
    )

    # TODO: Compute ADSIG
    # IF (ADELAY(I).GT.0.0D0)  THEN
    #   ADSIG(NEP,I) = DSQRT(COVF(I,I))
    # ELSE
    #   ADSIG(NEP,I) = 0.0D0       ! Unequivocal data flag
    # END IF

    # Reshape B_cov to be the same with COVF
    cm_shape = covariance_matrix.shape
    b_cov = b_cov[: cm_shape[0], : cm_shape[1]]

    # Compute COV_SD [COV_SD = B_COV * COVF * B_COV^T]
    covariance_std = b_cov @ covariance_matrix @ b_cov.T

    return np.linalg.inv(covariance_std)


def clean_zeros(input_array) -> np.ndarray:
    num_dims = len(input_array.shape)
    if num_dims == 1:
        # 1D array
        return np.array(np.trim_zeros(input_array))
    elif num_dims == 2:
        # 2D array
        return np.array(
            [np.trim_zeros(arr) for arr in input_array if np.trim_zeros(arr).size > 0]
        )
    else:
        raise ValueError("Only 1 or 2-D arrays are supported")


def calc_lsq_contrained(ATWA, ATWF, Q):
    """
    Performs least-squares estimation with linear constraints.
    """
    # Unconstrained
    X = scipy.optimize.lsq_linear(ATWA, ATWF).x

    # Constrained
    Zerr = 0 - (Q @ X)
    Zerr = clean_zeros(Zerr)

    MX = np.linalg.inv(ATWA)

    MXQT = MX @ Q.T
    MXQT = clean_zeros(MXQT)

    QMXQT = Q @ MXQT
    QMXQT = clean_zeros(QMXQT)

    QMX = Q @ MX
    QMX = clean_zeros(QMX)

    QMXQTIQMX = np.linalg.inv(QMXQT) @ QMX

    XDEL = clean_zeros(Zerr).T @ QMXQTIQMX
    XP = X + XDEL

    DELMX = MXQT @ QMXQTIQMX
    MXP = MX - DELMX

    return X, XP, MX, MXP
