from __future__ import annotations

from typing import Literal

import numba
import numpy as np
import pandas as pd
import scipy.stats
from nptyping import Float64, NDArray, Shape

from ..constants import SP_DEPTH, SP_SOUND_SPEED


def _compute_hm(svdf: pd.DataFrame, start_depth: float, end_depth: float) -> float:
    """
    Computes harmonic mean using `scipy's hmean <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hmean.html>`_ method.
    It takes the sound speed 'sv' as the input array and the depth 'dd' differences as the weights.

    Note that this function assumes absolute values for depth array,
    sound speed array, start depth, and end depth.

    The underlying formula is

    H = (w1+...+wn) / ((w1/x1)+...+(wn/xn))

    H is the resulting harmonic mean
    w is the weight value, in this case, the depth differences
    x is the input value, in this case, the sound speed

    Parameters
    ----------
    svdf : pd.DataFrame
        Sound speed profile data as dataframe with columns 'dd' and 'sv'
    start_depth : float
        The start depth for calculation
    end_depth : float
        The end depth for calculation

    """
    filtdf = svdf[(svdf[SP_DEPTH].round() >= start_depth) & (svdf[SP_DEPTH].round() <= end_depth)]

    # Get weights
    weights = filtdf[SP_DEPTH].diff()

    return scipy.stats.hmean(filtdf[SP_SOUND_SPEED], weights=weights, nan_policy="omit")


@numba.njit
def _sv_harmon_mean(
    dd: NDArray[Shape["*"], Float64],
    sv: NDArray[Shape["*"], Float64],
    zs: float,
    ze: float,
) -> float:
    """
    Compute harmonic mean of sound speed profile using algorithm from
    the original fortran implementation.

    Parameters
    ----------
    dd : (N,) ndarray
        The depth data.
        Values must be negative downward.
    sv : (N,) ndarray
        The sound speed data.
        Values must be positive.
    zs : float
        The start depth for harmonic mean to be computed
    ze : float
        The end depth for harmonic mean to be computed

    Returns
    -------
    float
        The sound speed harmonic mean result
    """
    # Ensure that depth and sound speed arrays
    # are the same shape
    assert (
        dd.shape == sv.shape
    ), f"dd and sv should have the same shape. dd:{dd.shape} != sv:{sv.shape}"

    zi = zs
    sum = 0.0

    for i in range(dd.shape[0]):
        z1, c_z1 = dd[i], sv[i]
        z2, c_z2 = dd[i + 1], sv[i + 1]

        if i > 0:
            zi = z1
        if z1 <= ze:
            break

        zf = z2
        if z2 < ze:
            zf = ze

        # Compute slope
        b = (c_z2 - c_z1) / (z2 - z1)

        # Ensure that slope is not 0
        assert b != 0.0, "Slope is zero"

        # Compute the weight
        wi = np.log((zi - z1) * b + c_z1) / b
        wf = np.log((zf - z1) * b + c_z1) / b
        w_d = wf - wi
        sum += w_d
    return (ze - zs) / sum


def sv_harmonic_mean(
    svdf: pd.DataFrame,
    start_depth: float,
    end_depth: float,
    method: Literal["scipy", "numba"] = "numba",
) -> float:
    """
    Computes harmonic mean from a sound profile
    containing depth (dd) and sound speed (sv)

    Parameters
    ----------
    svdf : pd.DataFrame
        Sound speed profile data as dataframe
    start_depth : int or float
        The start depth for harmonic mean to be computed
    end_depth : int or float
        The end depth for harmonic mean to be computed
    method : {"scipy", "numba"}, default "numba"
        The method to use for computing the harmonic mean.
        The options are "scipy" and "numba".

        "scipy" uses scipy.stats.hmean method.
        "numba" uses numba.njit method and it's an equivalent algorithm,
        to the original fortran implementation.

    Returns
    -------
    float
        The sound speed harmonic mean value
    """
    if svdf.empty:
        msg: str = "Dataframe is empty! Please check your data inputs."
        raise ValueError(msg)
    # Clean up the sound speed value, ensuring that there's no negative value
    svdf = svdf[svdf[SP_SOUND_SPEED] > 0].reset_index(drop=True)

    for col in [SP_DEPTH, SP_SOUND_SPEED]:
        if col not in svdf.columns:
            msg: str = f"{col} column must exist in the input dataframe!"
            raise ValueError(msg)

    # lower the strings to normalize input
    method = method.lower()
    if method == "scipy":
        # Make all of the values absolute values, so we're only dealing with positives
        abs_start = abs(start_depth)
        abs_end = abs(end_depth)
        abs_sv = abs(svdf)

        svhm = _compute_hm(abs_sv, abs_start, abs_end)
    elif method == "numba":
        # Ensure both start and end depths are negative
        start_depth = -np.abs(start_depth)
        end_depth = -np.abs(end_depth)

        if start_depth < end_depth:
            msg: str = f"Start depth {start_depth} must be greater than end depth {end_depth}!"
            raise ValueError(msg)

        svdf = svdf[(svdf[SP_DEPTH].round() <= start_depth)]
        # Extract the numpy arrays for depth and sound speed
        # Make sure that the depth array has negative values
        # and the sound speed array has positive values
        dd = -np.abs(svdf[SP_DEPTH].values)
        sv = np.abs(svdf[SP_SOUND_SPEED].values)

        svhm = _sv_harmon_mean(dd, sv, start_depth, end_depth)
    else:
        msg = f"Method {method} is not implemented!"
        raise NotImplementedError(msg)

    return svhm
