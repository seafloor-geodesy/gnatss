from __future__ import annotations

import numba
import numpy as np
from nptyping import Float64, NDArray, Shape


@numba.njit(cache=True)
def simple_twtt(
    transmit_vectors: NDArray[Shape["*, 3"], Float64],
    reply_vectors: NDArray[Shape["*, 3"], Float64],
    transponders_mean_sv: NDArray[Shape["*"], Float64],
) -> NDArray[Shape["*"], Float64]:
    """
    Calculate the Simple Modeled TWTT (Two way travel time) in seconds

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
    transmit_distance = np.array([np.linalg.norm(vector) for vector in transmit_vectors])
    reply_distance = np.array([np.linalg.norm(vector) for vector in reply_vectors])

    return (transmit_distance + reply_distance) / transponders_mean_sv
