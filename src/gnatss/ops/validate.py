from typing import Tuple

import numba
import numpy as np
import pandas as pd
import scipy
from nptyping import Float64, NDArray, Shape
from numba.typed import List as NumbaList

from .. import constants
from . import calc_std_and_verify
from .utils import clean_zeros

# Constrain matrix Q
# Hardcoded constant for 3 transponders
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


@numba.njit(cache=True)
def _calc_qmxqt(atwa: NDArray, q: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    mx = np.linalg.inv(atwa)

    mxqt = mx @ q.T
    qmxqt = q @ mxqt
    qmx = q @ mx

    return qmxqt, qmx, mx, mxqt


@numba.njit(cache=True)
def _calc_xp(x: NDArray, qmxqtiqmx: NDArray, zerr: NDArray) -> NDArray:
    xdel = zerr.T @ qmxqtiqmx
    xp = x + xdel
    return xp


@numba.njit(cache=True)
def _calc_mxp(mx: NDArray, qmxqtiqmx: NDArray, mxqt: NDArray) -> NDArray:
    delmx = mxqt @ qmxqtiqmx
    mxp = mx - delmx
    return mxp


@numba.njit(cache=True)
def _sum_all(input_list):
    total = np.zeros_like(input_list[0])
    for arr in input_list:
        total += arr
    return total


@numba.njit(cache=True)
def _create_q_matrix(num_transponders: int) -> NDArray:
    mpars = 3 * num_transponders
    q = np.zeros(shape=(mpars, mpars))
    neg = 0
    pos = 3
    for i in range(0, mpars):
        if pos < mpars:
            q[i, neg] = -1.0
            q[i, pos] = 1.0
            neg += 1
            pos += 1
    return q


@numba.njit(cache=True)
def _combine_results(all_results):
    all_atwa = NumbaList()
    all_atwf = NumbaList()
    travel_time_residuals = NumbaList()
    sigma_delays = NumbaList()

    for atwa, atwf, tt_residual, sigma_delay in all_results:
        all_atwa.append(atwa)
        all_atwf.append(atwf)
        travel_time_residuals.append(tt_residual)
        sigma_delays.append(sigma_delay)

    return all_atwa, all_atwf, travel_time_residuals, sigma_delays


def calc_lsq_constrained(
    atwa: NDArray[Shape["*, *"], Float64],
    atwf: NDArray[Shape["*"], Float64],
    num_transponders: int,
):
    """
    Performs least-squares estimation with linear constraints.
    This function has been translated almost directly from ``lscd2.f`` code.

    Parameters
    ----------
    atwa : ndarray
        ATWA matrix from (A partials matrix)^T * Weight matrix * A partials matrix
    atwf : ndarray
        ATWF matrix from (A partials matrix)^T * Weight matrix * Travel time residuals
    num_transponders : int
        The number of transponders

    Returns
    -------
    x : (N,) ndarray
        Solution vector without constraints
    xp : (N,) ndarray
        Solution vector with constraints
    mx : (N,N) ndarray
        Covariance matrix of solution without constraints
    mxp : (N,N) ndarray
        Covariance matrix of solution with constraints
    """
    # Setup Q Matrix
    q = _create_q_matrix(num_transponders)

    # Unconstrained
    x = scipy.optimize.lsq_linear(atwa, atwf).x

    # Constrained
    zerr = 0 - (q @ x)
    qmxqt, qmx, mx, mxqt = _calc_qmxqt(atwa, q)

    # Clean up zeros
    qmxqt = clean_zeros(qmxqt)
    qmx = clean_zeros(qmx)
    mx = clean_zeros(mx)
    mxqt = clean_zeros(mxqt)
    zerr = clean_zeros(zerr)

    qmxqtiqmx = np.linalg.inv(qmxqt) @ qmx

    # Get constrained solution and its covariance
    xp = _calc_xp(x, qmxqtiqmx, zerr)
    mxp = _calc_mxp(mx, qmxqtiqmx, mxqt)

    return (
        x,  # Unconstrained solution
        xp,  # Unconstrained covariance
        mx,  # Constrained solution
        mxp,  # Constrained covariance
    )


def check_solutions(all_results, transponders_xyz):
    num_transponders = len(transponders_xyz)
    all_atwa, all_atwf, travel_time_residuals, sigma_delays = _combine_results(
        all_results
    )
    atwa = _sum_all(all_atwa)
    atwf = _sum_all(all_atwf)
    (
        dp0,  # Unconstrained solution
        delp,  # Unconstrained covariance
        cp0,  # Constrained solution
        covpx,  # Constrained covariance
    ) = calc_lsq_constrained(atwa, atwf, num_transponders)

    # Retrieve diagonal and change negative values to 0
    diag = covpx.diagonal().copy()
    diag[diag < 0] = 0

    # Compute sigpx
    sigpx = np.sqrt(diag)

    # Test convergence
    is_converged = np.all(np.logical_or((np.abs(delp) <= (0.1 * sigpx)), (sigpx == 0)))

    if not is_converged:
        transponders_xyz = transponders_xyz + delp[:3]

    check_result = {
        "dp0": dp0,
        "delp": delp,
        "cp0": cp0,
        "covpx": covpx,
        "sigpx": sigpx,
        "address": travel_time_residuals,
        "adsig": sigma_delays,
    }

    return is_converged, transponders_xyz, check_result


def check_sig3d(data: pd.DataFrame, gps_sigma_limit: float):
    # Compute 3d standard deviation
    data[constants.SIG_3D] = data.apply(
        calc_std_and_verify, axis="columns", verify=False
    )

    # TODO: Put debug for the times that are not valid and option to save to file
    # In fortran it get sent to GPS_3drms_exceeds
    # Find a way to distinguish between transmit and reply
    # data_exceeds = data[data.sig_3d > gps_sigma_limit]

    # Filter out everything that exceeds the gps sigma limit
    data = data[data.sig_3d < gps_sigma_limit]
    return data.drop(constants.SIG_3D, axis="columns")