from typing import Tuple

import numba
import numpy as np
from nptyping import Float64, NDArray, Shape
from numba.typed import List as NumbaList

from .utils import calc_uv


@numba.njit(cache=True)
def _calc_tr_vectors(
    transponders_xyz: NDArray[Shape["*, 3"], Float64],  # noqa
    transmit_xyz: NDArray[Shape["3"], Float64],
    reply_xyz: NDArray[Shape["*, 3"], Float64],  # noqa
) -> Tuple[NDArray[Shape["*, 3"]], NDArray[Shape["*, 3"]]]:  # noqa
    """
    Calculate the transmit and reply vectors

    Parameters
    ----------
    transponders_xyz : (N,3) ndarray
        The transponders xyz locations
    transmit_xyz : (3,) ndarray
        The transmit xyz location
    reply_xyz : (N,3) ndarray
        The reply xyz locations

    Returns
    -------
    transmit_vectors : (N,3) ndarray
        The transmit array of vectors
    reply_vectors : (N,3) ndarray
        The reply array of vectors
    """
    transmit_vectors = transponders_xyz - transmit_xyz
    reply_vectors = transponders_xyz - reply_xyz
    return transmit_vectors, reply_vectors


@numba.njit(cache=True)
def _calc_unit_vectors(
    vectors: NDArray[Shape["*, 3"], Float64]
) -> NDArray[Shape["*, 3"], Float64]:
    """
    Calculates the unit vectors from an array of vectors.
    This function will go through each vector array,
    and compute its unit vector.
    """
    u_vectors = np.empty_like(vectors)
    n = len(vectors)
    for i in range(n):
        u_vectors[i] = calc_uv(vectors[i])
    return u_vectors


@numba.njit(cache=True)
def _calc_partial_derivatives(
    transmit_uv: NDArray[Shape["*, 3"], Float64],
    reply_uv: NDArray[Shape["*, 3"], Float64],
    transponders_mean_sv: NDArray[Shape["*"], Float64],
) -> NDArray[Shape["*, 3"], Float64]:
    """
    Calculates the partial derivative

    .. math::
        \\frac{\\hat{D_s} + \\hat{D_r}}{c}

    Parameters
    ----------
    transmit_uv : (3,N) ndarray
        The transmit array of unit vectors
    reply_uv : (3,N) ndarray
        The reply array of unit vectors

    Returns
    -------
    (3,N) ndarray
        The resulting partial derivatives matrix
    """
    return (transmit_uv + reply_uv) / transponders_mean_sv


@numba.njit(cache=True)
def _setup_ab(delays, num_transponders, partial_derivatives):
    """
    Setup a partials and b cov
    """
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
    return A_partials, B_cov


@numba.njit(cache=True)
def _calc_cov(
    transmit_uv, gps_covariance_matrix, travel_times_variance, transponders_mean_sv
):
    """
    Calculate the weight factor array
    """
    # Calculate covariance matrix for partlp vectors (COVF) Units m^2
    covariance_matrix = np.abs((transmit_uv @ gps_covariance_matrix @ transmit_uv.T))

    # Weighting factor equal to 2/sv_mean
    weight_factor = _calc_weight_fac(transponders_mean_sv)

    # Convert to covariance of acoustic delays Units seconds^2
    covariance_matrix = weight_factor * covariance_matrix

    # Add travel times variance on diagonal only
    n = len(covariance_matrix)
    for i in range(n):
        covariance_matrix[i, i] = covariance_matrix[i, i] + travel_times_variance
    return covariance_matrix


@numba.njit(cache=True)
def _calc_weight_fac(transponders_mean_sv):
    """
    Calculate the weight factor array

    .. math::
        W_f = \\frac{2.0}{s^2}
    """
    return 2.0 / (transponders_mean_sv**2)


@numba.njit(cache=True)
def _calc_weight_mat(covariance_std):
    """
    Calculate the weight matrix by inverting the covariance std matrix
    """
    return np.linalg.inv(covariance_std)


@numba.njit(cache=True)
def __get_diagonal(array: NDArray[Shape["*, *"], Float64]) -> NDArray:
    """
    Get the diagonal of an array

    Parameters
    ----------
    array : (N,M) ndarray
        The input array to get diagonal from,
        array shape must be even

    Returns
    -------
    (N,) ndarray
        The diagonal values of the input array
    """
    len_cm = len(array)
    diag = np.zeros(len_cm)
    for i in range(len_cm):
        diag[i] = array[i, i]
    return diag


@numba.njit(cache=True)
def calc_twtt_model(
    transmit_vectors: NDArray[Shape["*, 3"], Float64],
    reply_vectors: NDArray[Shape["*, 3"], Float64],
    transponders_mean_sv: NDArray[Shape["*"], Float64],
) -> NDArray[Shape["*"], Float64]:
    """
    Calculate the Modeled TWTT (Two way travel time) in seconds

    .. math::
        \\frac{\\hat{D_s} + \\hat{D_r}}{c}

    Parameters
    ----------
    transmit_vector : (N,3) ndarray
        The transmit array of vectors
    reply_vector : (N,3) ndarray
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


@numba.njit(cache=True)
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


@numba.njit(cache=True)
def solve_transponder_locations(
    transmit_xyz: NDArray,
    reply_xyz: NDArray,
    gps_covariance_matrix: NDArray,
    observed_delays: NDArray,
    transponders_xyz: NDArray,
    transponders_delay: NDArray,
    transponders_mean_sv: NDArray,
    travel_times_variance: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Solve transponder locations by performing basic GNSS-Acoustic
    Derivation.
    """
    # Infer the number of transponders by the number of mean sv
    num_transponders = len(transponders_mean_sv)

    # Get the transmit and reply vectors from xyz locations
    transmit_vectors, reply_vectors = _calc_tr_vectors(
        transponders_xyz, transmit_xyz, reply_xyz
    )

    # Calculate Modeled TWTT (Two way travel time) in seconds
    twtt_model = calc_twtt_model(transmit_vectors, reply_vectors, transponders_mean_sv)

    # Calculate the travel time residual
    tt_residual = calc_tt_residual(observed_delays, transponders_delay, twtt_model)

    # Calculate the unit vectors
    transmit_uv = _calc_unit_vectors(transmit_vectors)
    reply_uv = _calc_unit_vectors(reply_vectors)

    # Calculate the partial derivatives
    partial_derivatives = _calc_partial_derivatives(
        transmit_uv, reply_uv, transponders_mean_sv
    )

    # Setup the A partial derivative matrix and B covariance matrix
    a_partials, b_cov = _setup_ab(
        observed_delays, num_transponders, partial_derivatives
    )

    # Calculate covariance matrix for partlp vectors (COVF) Units m^2
    covariance_matrix = _calc_cov(
        transmit_uv, gps_covariance_matrix, travel_times_variance, transponders_mean_sv
    )

    # Get the array diagonal and compute the sigma values
    diag_cov = __get_diagonal(covariance_matrix)
    sigma_delay = np.sqrt(diag_cov)

    # Reshape B_cov to be the same with COVF
    cm_shape = covariance_matrix.shape
    b_cov = np.ascontiguousarray(b_cov[: cm_shape[0], : cm_shape[1]])

    # Calculate the covariance standard deviation
    covariance_std = b_cov @ covariance_matrix @ b_cov.T

    # Calculate the weight matrix from covariance standard deviation
    weight_matrix = _calc_weight_mat(covariance_std)

    # Perform inversion
    atwa = a_partials.T @ weight_matrix @ a_partials
    atwf = a_partials.T @ weight_matrix @ tt_residual

    return atwa, atwf, tt_residual, sigma_delay


@numba.njit(cache=True)
def perform_solve(
    data_inputs,
    transponders_mean_sv,
    transponders_xyz,
    transponders_delay,
    travel_times_variance,
):
    all_results = NumbaList()
    for transmit_xyz, reply_xyz, gps_covariance_matrix, observed_delays in data_inputs:
        results = solve_transponder_locations(
            transmit_xyz,
            reply_xyz,
            gps_covariance_matrix,
            observed_delays,
            transponders_xyz,
            transponders_delay,
            transponders_mean_sv,
            travel_times_variance,
        )
        all_results.append(results)
    return all_results
