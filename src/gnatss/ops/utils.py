from typing import Any, List

import numba
import numpy as np
from nptyping import NDArray, Shape

DEFAULT_VECTOR_NORM = np.array([2.0, 0.0, 0.0])


# Numba utilities


@numba.njit(cache=True)
def calc_uv(input_vector: NDArray[Shape["3"], Any]) -> NDArray[Shape["3"], Any]:
    """
    Calculate unit vector for a 1-D input vector of size 3

    Parameters
    ----------
    input_vector : (3,) ndarray
        A 1-D input vector as numpy array

    Returns
    -------
    (3,) ndarray
        The resulting unit vector as numpy array

    Raises
    ------
    ValueError
        If the input vector is not a 1-D array
    """

    if input_vector.ndim != 1 or input_vector.shape != (3,):
        raise ValueError("Unit vector calculation must be 1-D array of shape 3!")

    vector_norm = np.linalg.norm(input_vector)

    if vector_norm == 0:
        return DEFAULT_VECTOR_NORM

    return input_vector / vector_norm


# Regular Python Utilities
def _prep_col_names(col_names: List[str], transmit: bool = True) -> List[str]:
    """Prepare column names for either transmission or reply by adding 0 or 1"""
    suffix = "0" if transmit else "1"
    return [name + suffix for name in col_names]


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
