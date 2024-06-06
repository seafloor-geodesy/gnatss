from __future__ import annotations

from typing import Any

import numba
import numpy as np
from nptyping import NDArray, Shape

DEFAULT_VECTOR_NORM = np.array([2.0, 0.0, 0.0])


# Numba utilities


@numba.njit(cache=True)
def calc_uv(input_vector: NDArray[Shape[3], Any]) -> NDArray[Shape[3], Any]:
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
    ashape = input_vector.shape

    # Dimensionality check already done by numba
    # so we just check for the shape
    assert ashape == (3,), (
        "Unit vector calculation must be 1-D array of shape 3! "
        f"Instead got 1-D of shape {','.join([str(s) for s in ashape])}!"
    )

    vector_norm = np.linalg.norm(input_vector)

    if vector_norm == 0:
        return DEFAULT_VECTOR_NORM

    return input_vector / vector_norm


# Regular Python Utilities
def _prep_col_names(col_names: list[str], transmit: bool = True) -> list[str]:
    """
    Prepares column names for either transmit
    or reply by adding 0 or 1

    Parameters
    ----------
    col_names : list[str]
        A list of column names to be modified
    transmit : bool, optional
        Flag to signify modification
        for transmit names, by default True

    Returns
    -------
    list[str]
        The modified list of column names
    """
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
    if num_dims == 2:
        # 2D array
        return np.array([np.trim_zeros(arr) for arr in input_array if np.trim_zeros(arr).size > 0])

    err_msg = f"Only 1 or 2-D arrays are supported, instead for {num_dims} dimensions"
    raise NotImplementedError(err_msg)
