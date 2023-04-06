import math
from typing import Union

import numba
import numpy as np
import pandas as pd


@numba.njit
def _compute_hm(
    dd: np.ndarray,
    sv: np.ndarray,
    start_depth: Union[int, float],
    end_depth: Union[int, float],
    start_index: int,
):
    """
    Computes harmonic mean.
    It's a direct translation from the original Fortran code found in
    src/cal_sv_harmonic_mean/get_sv_harmonic_mean.F called
    subroutine `sv_harmon_mean`
    """
    # TODO: Find a way to vectorize this computation

    # Assign start depth and end depth to zs and ze
    zs = start_depth
    ze = end_depth

    # Extract the first and second depth values
    z1 = dd[start_index]
    z2 = dd[start_index + 1]

    # Extract the first and second sound speed values
    c_z1 = sv[start_index]
    c_z2 = sv[start_index + 1]

    # Set start depth to initial depth to keep track of it
    zi = zs

    # If the second depth value (z2) is greater than and equal to the
    # end depth value (ze) then the final depth (zf) should be assigned
    # to the end depth value (ze), otherwise, the final depth
    # should be assigned to the second depth values
    if z2 >= ze:
        zf = ze
    else:
        zf = z2

    # Start cumulative sum as 0.0
    cumsum = 0.0

    # Loop over the whole array for depth and sound speed,
    # starting from the index of the second value to the whole
    # depth data, which is assumed to be the same exact number
    # as the sound speed data
    for i in range(start_index + 1, len(dd)):
        # calculate the slope of the two points
        # for sound speed and depth
        # slope = (sv2 - sv1) / (d2 - d1)
        b = (c_z2 - c_z1) / (z2 - z1)
        wi = zi - z1 + c_z1 / b
        wf = zf - z1 + c_z1 / b

        wi = math.log((zi - z1) * b + c_z1) / b
        wf = math.log((zf - z1) * b + c_z1) / b

        delta = wf - wi
        cumsum = cumsum + delta
        z1 = zf
        z2 = dd[i + 1]
        c_z1 = c_z2
        c_z2 = sv[i + 1]
        zi = zf

        if ze > zi and ze < z2:
            zf = ze
        else:
            zf = z2

        if z1 >= ze:
            break

    if cumsum == 0:
        # If cumulative sum is 0, most likely only one value
        return sv[start_index]
    return (ze - zs) / cumsum


def sv_harmonic_mean(
    svdf: pd.DataFrame, start_depth: Union[int, float], end_depth: Union[int, float]
):
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

    Returns
    -------
    float
        The sound speed harmonic mean value
    """
    # Clean up the sound speed value, ensuring that there's no negative value
    svdf = svdf[svdf["sv"] > 0].reset_index(drop=True)
    # Make all of the values absolute values, so we're only dealing with positives
    abs_start = abs(start_depth)
    abs_end = abs(end_depth)
    abs_sv = abs(svdf)
    # Get the index for the start of depth closest to specified start depth
    try:
        start_index = abs_sv[(abs_sv["dd"].round() >= abs_start)].index[0]
    except IndexError:
        raise ValueError("Dataframe is empty! Please check your data inputs.")

    return _compute_hm(
        abs_sv["dd"].values, abs_sv["sv"].values, abs_start, abs_end, start_index
    )
