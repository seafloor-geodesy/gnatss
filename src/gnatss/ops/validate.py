from __future__ import annotations

import itertools
from typing import Literal

import numba
import numpy as np
import pandas as pd
import scipy
from nptyping import Float64, NDArray, Shape
from numba.typed import List as NumbaList

from .. import constants
from .utils import clean_zeros


@numba.njit(cache=True)
def _calc_qmxqt(atwa: NDArray, q: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    mx = np.linalg.inv(atwa)

    mxqt = mx @ q.T
    qmxqt = q @ mxqt
    qmx = q @ mx

    return qmxqt, qmx, mx, mxqt


@numba.njit(cache=True)
def _calc_xp(x: NDArray, qmxqtiqmx: NDArray, zerr: NDArray) -> NDArray:
    xdel = zerr.T @ qmxqtiqmx
    return x + xdel


@numba.njit(cache=True)
def _calc_mxp(mx: NDArray, qmxqtiqmx: NDArray, mxqt: NDArray) -> NDArray:
    delmx = mxqt @ qmxqtiqmx
    return mx - delmx


@numba.njit(cache=True)
def _sum_all(input_list: NumbaList) -> NDArray:
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
    for i in range(mpars):
        if pos < mpars:
            q[i, neg] = -1.0
            q[i, pos] = 1.0
            neg += 1
            pos += 1
    return q


@numba.njit(cache=True)
def _combine_results(
    all_results: NumbaList,
) -> tuple[NumbaList, NumbaList, NumbaList, NumbaList]:
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
    all_atwa, all_atwf, travel_time_residuals, sigma_delays = _combine_results(all_results)
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


def _check_cols_in_series(input_series: pd.Series, columns: list[str]) -> None:
    """Private func to check if the columns exists in data series"""
    for col in columns:
        if col not in input_series:
            # Catch if some of thecolumns are missing
            msg: str = f"{col} not found in the data series provided."
            raise KeyError(msg)


def calc_std_and_verify(
    gps_series: pd.Series,
    std_dev: bool = True,
    sigma_limit: float = 0.05,
    verify=True,
    data_type: Literal["receive", "transmit"] = "receive",
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
    # Compute the 3d std (sum variances of GPS components and take sqrt)
    sig_3d = np.sqrt(np.sum(gps_series ** (2 if std_dev else 1)))

    if verify and (sig_3d > sigma_limit):
        # Verify sigma value, throw error if greater than gps sigma limit
        msg = f"3D Standard Deviation of {sig_3d} exceeds " f"GPS Sigma Limit of {sigma_limit}!"
        raise ValueError(msg)

    return sig_3d


def check_sig3d(data: pd.DataFrame, gps_sigma_limit: float):
    # Get covariance diagonal columns
    diag_cov_cols = {
        "receive": [*constants.DATA_SPEC.gnss_rx_diag_cov_fields.keys()],
        "transmit": [*constants.DATA_SPEC.gnss_tx_diag_cov_fields.keys()],
    }
    diag_cov_columns = list(itertools.chain.from_iterable(diag_cov_cols.values()))
    # Checks for GPS Covariance Diagonal values
    assert all(col in data.columns for col in diag_cov_columns)

    # Compute 3d standard deviation
    rx_sig_3d = data[diag_cov_cols["receive"]].apply(
        calc_std_and_verify, axis="columns", verify=False, data_type="receive"
    )
    # Filter out everything that exceeds the gps sigma limit
    data = data[rx_sig_3d < gps_sigma_limit]

    tx_sig_3d = data[diag_cov_cols["transmit"]].apply(
        calc_std_and_verify, axis="columns", verify=False, data_type="transmit"
    )

    # TODO: Put debug for the times that are not valid and option to save to file
    # In fortran it get sent to GPS_3drms_exceeds
    # Find a way to distinguish between transmit and reply
    # data_exceeds = data[data.sig_3d > gps_sigma_limit]

    # Filter out everything that exceeds the gps sigma limit
    return data[tx_sig_3d < gps_sigma_limit]
