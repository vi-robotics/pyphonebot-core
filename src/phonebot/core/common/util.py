#!/usr/bin/env python3

"""
Utility Functions.
"""

import time
from typing import Any, Optional
import numpy as np

__all__ = ['find_nearest_index', 'find_k_nearest_indices']


time_scale = 0.5


def increment_time(dt=0.1):
    pass


def get_time():
    return time.time() * time_scale


def get_time_scale():
    return time_scale


def find_nearest_index(array: np.ndarray, value: Any) -> int:
    """
    Find the index of the element nearest to value in an array.

    https://stackoverflow.com/a/26026189


    Args:
        array (np.ndarray): The array to query
        value (Any): The value to search for in the array.

    Returns:
        int: The index of the nearest value.
    """
    idx = np.searchsorted(array, value, side="left")
    is_left = idx > 0 and (idx == len(array) or np.abs(
        value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx - is_left


def find_k_nearest_indices(array: np.ndarray,
                           value: Any,
                           k: int = 1) -> Optional[np.ndarray]:
    """Find k nearest indices in an array, sorted by proximity.

    Args:
        array (np.ndarray): The array to find nearest indices in.
        value (Any): The value in the array to find the nearest indices
            to.
        k (int, optional): The number of indices to return. Defaults to 1.

    Returns:
        Optional[np.ndarray]: An #K vector of indices, or None if there are
            fewer than k elements in the array.
    """
    n = len(array)
    if n < k:
        return None
    anchor = find_nearest_index(array, value)
    indices = [anchor]

    prv_index = anchor - 1
    nxt_index = anchor + 1
    while len(indices) < k:
        error_prv = np.abs(array[prv_index] - value)
        error_nxt = np.abs(array[nxt_index] -
                           value) if (nxt_index < n) else np.inf
        if error_prv < error_nxt:
            indices.append(prv_index)
            prv_index -= 1
        else:
            indices.append(nxt_index)
            nxt_index += 1
    return np.asarray(indices)
