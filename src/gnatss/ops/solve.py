import numba
import numpy as np
from nptyping import Float64, NDArray, Shape
from numba.typed import List as NumbaList

from .utils import calc_uv


@numba.njit
def _calc_tr_vectors(transponders_xyz, transmit_xyz, reply_xyz):
    """Calculate transmit and reply vector"""
    transmit_vectors = transponders_xyz - transmit_xyz
    reply_vectors = transponders_xyz - reply_xyz
    return transmit_vectors, reply_vectors


@numba.njit
def _calc_unit_vectors(vectors):
    u_vectors = np.empty_like(vectors)
    n = len(vectors)
    for i in range(n):
        u_vectors[i] = calc_uv(vectors[i])
    return u_vectors


@numba.njit
def _calc_partial_derivatives(transmit_uv, reply_uv, transponders_mean_sv):
    return (transmit_uv + reply_uv) / transponders_mean_sv


@numba.njit
def _setup_ab(delays, num_transponders, partial_derivatives):
    """Setup a partials and b cov"""
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


@numba.njit
def _calc_cov(
    transmit_uv, gps_covariance_matrix, travel_times_variance, transponders_mean_sv
):
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


@numba.njit
def _calc_weight_fac(transponders_mean_sv):
    return 2.0 / (transponders_mean_sv**2)


@numba.njit
def _calc_weight_mat(covariance_std):
    return np.linalg.inv(covariance_std)


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


@numba.njit
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


@numba.njit
def solve_transponder_locations(
    transmit_xyz,
    reply_xyz,
    gps_covariance_matrix,
    observed_delays,
    transponders_xyz,
    transponders_delay,
    transponders_mean_sv,
    travel_times_variance,
    num_transponders,
):
    transmit_vectors, reply_vectors = _calc_tr_vectors(
        transponders_xyz, transmit_xyz, reply_xyz
    )

    # Calculate Modeled TWTT (Two way travel time) in seconds
    twtt_model = calc_twtt_model(transmit_vectors, reply_vectors, transponders_mean_sv)

    # Calculate the travel time residual
    tt_residual = calc_tt_residual(observed_delays, transponders_delay, twtt_model)

    transmit_uv = _calc_unit_vectors(transmit_vectors)
    reply_uv = _calc_unit_vectors(reply_vectors)

    # Calculate the partial derivatives
    partial_derivatives = _calc_partial_derivatives(
        transmit_uv, reply_uv, transponders_mean_sv
    )

    a_partials, b_cov = _setup_ab(
        observed_delays, num_transponders, partial_derivatives
    )

    # Calculate covariance matrix for partlp vectors (COVF) Units m^2
    covariance_matrix = _calc_cov(
        transmit_uv, gps_covariance_matrix, travel_times_variance, transponders_mean_sv
    )

    len_cm = len(covariance_matrix)
    diag_cov = np.zeros(len_cm)
    for i in range(len_cm):
        diag_cov[i] = covariance_matrix[i, i]
    sigma_delay = np.sqrt(diag_cov)

    # Reshape B_cov to be the same with COVF
    cm_shape = covariance_matrix.shape
    b_cov = np.ascontiguousarray(b_cov[: cm_shape[0], : cm_shape[1]])

    covariance_std = b_cov @ covariance_matrix @ b_cov.T

    weight_matrix = _calc_weight_mat(covariance_std)

    # Perform inversion
    atwa = a_partials.T @ weight_matrix @ a_partials
    atwf = a_partials.T @ weight_matrix @ tt_residual

    return atwa, atwf, tt_residual, sigma_delay


@numba.njit
def _perform_solve(
    data_inputs,
    transponders_mean_sv,
    transponders_xyz,
    transponders_delay,
    num_transponders,
):
    all_results = NumbaList()
    for data in data_inputs:
        (
            transmit_xyz,
            reply_xyz,
            gps_covariance_matrix,
            observed_delays,
            travel_times_variance,
        ) = data
        results = solve_transponder_locations(
            transmit_xyz,
            reply_xyz,
            gps_covariance_matrix,
            observed_delays,
            transponders_xyz,
            transponders_delay,
            transponders_mean_sv,
            travel_times_variance,
            num_transponders,
        )
        all_results.append(results)
    return all_results
