from typing import List, Tuple

import numpy as np
import pandas as pd
from nptyping import Float64, NDArray, Shape

from .. import constants
from .utils import calc_uv

__all__ = ["calc_uv"]

# Constrain matrix Q
Q_MATRIX = np.array(
    [
        [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


def _check_cols_in_series(input_series: pd.Series, columns: List[str]) -> None:
    """Private func to check if the columns exists in data series"""
    for col in columns:
        if col not in input_series:
            # Catch if some of thecolumns are missing
            raise KeyError(f"``{col}`` not found in the data series provided.")


def calc_std_and_verify(
    gps_series: pd.Series, std_dev: bool = True, sigma_limit: float = 0.05, verify=True
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
    verify : bool
        Flag to run verification or not

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
    _check_cols_in_series(input_series=gps_series, columns=constants.GPS_COV_DIAG)

    # Compute the 3d std (sum variances of GPS components and take sqrt)
    sig_3d = np.sqrt(
        np.sum(gps_series[constants.GPS_COV_DIAG] ** (2 if std_dev else 1))
    )

    if verify and (sig_3d > sigma_limit):
        # Verify sigma value, throw error if greater than gps sigma limit
        raise ValueError(
            f"3D Standard Deviation of {sig_3d} exceeds GPS Sigma Limit of {sigma_limit}!"
        )

    return sig_3d


def calc_partials(
    transmit_vectors: NDArray[Shape["3, *"], Float64],
    reply_vectors: NDArray[Shape["3, *"], Float64],
    transponders_mean_sv: NDArray[Shape["*"], Float64],
    delays: NDArray[Shape["*"], Float64],
    num_transponders: int,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Calculate the partial derivative matrices

    Parameters
    ----------
    transmit_vectors : (3,N) ndarray
        The transmit array of vectors
    reply_vectors : (3,N) ndarray
        The reply array of vectors
    transponders_mean_sv : (N,) ndarray
        The transponders mean sound speed
    delays : (N,) ndarray
        The measured travel time delays from data (sec)
    num_transponders : int
        The total number of transponders

    Returns
    -------
    A_partials : (num_transponders,num_transponders*3) ndarray
        The A partial derivatives matrix
    B_cov : (num_transponders,num_transponders*3) ndarray
        The B covariance matrix
    transmit_uv : (3,N) ndarray
        The transmit array of unit vectors
    reply_uv : (3,N) ndarray
        The reply array of unit vectors
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
    transmit_uv: NDArray[Shape["3, *"], Float64],
    gps_covariance_matrix: NDArray[Shape["3, 3"], Float64],
    transponders_mean_sv: NDArray[Shape["*"], Float64],
    b_cov: NDArray,
    travel_times_variance: float,
):
    """
    Calculate the weight matrix

    Parameters
    ----------
    transmit_uv : (3,N) ndarray
        The transmit array of unit vectors
    gps_covariance_matrix : (3,3) ndarray
        The covariance matrix from the gps data.
        These are the values from "xx,xy,xz,yx,yy,yz,zx,zy,zz"
    transponders_mean_sv : (N,) ndarray
        The transponders mean sound speed
    b_cov : (num_transponders,num_transponders*3) ndarray
        The B covariance matrix
    travel_times_variance : float
        The user specified travel time variance value

    Returns
    -------
    (num_transponders,num_transponders) ndarray
        The resulting weight matrix
    ndarray
        The sigma delay array
    """
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

    # Compute delay sigma
    sigma_delay = np.sqrt(covariance_matrix.diagonal())

    # Reshape B_cov to be the same with COVF
    cm_shape = covariance_matrix.shape
    b_cov = b_cov[: cm_shape[0], : cm_shape[1]]

    # Compute COV_SD [COV_SD = B_COV * COVF * B_COV^T]
    covariance_std = b_cov @ covariance_matrix @ b_cov.T

    return np.linalg.inv(covariance_std), sigma_delay


def clean_zeros(input_array: NDArray) -> NDArray:
    """
    Trim the leading and/or trailing zeros from a 1-D or 2-D arrays.

    Parameters
    ----------
    input_array : (N,) ndarray or (N,N) ndarray

    Returns
    -------
    ndarray
        The resulting N-D array with leading or trailing zeroes trimmed

    Raises
    ------
    NotImplementedError
        If the ``input_array`` not a 1 or 2-D array
    """
    num_dims = len(input_array.shape)
    if num_dims == 1:
        # 1D array
        return np.array(np.trim_zeros(input_array))
    elif num_dims == 2:
        # 2D array
        return np.array(
            [np.trim_zeros(arr) for arr in input_array if np.trim_zeros(arr).size > 0]
        )

    raise NotImplementedError(
        f"Only 1 or 2-D arrays are supported, instead for {num_dims} dimensions"
    )
