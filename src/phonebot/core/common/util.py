#!/usr/bin/env python3

"""
Utility Functions.
"""

import time
import numpy as np

__all__ = ['find_nearest_index', 'find_k_nearest_indices']


# phonebot_time = 0.0
# def increment_time(dt=0.1):
#    global phonebot_time
#    phonebot_time += dt
#
#
# def get_time():
#    global phonebot_time
#    return phonebot_time


time_scale = 0.5


def increment_time(dt=0.1):
    pass


def get_time():
    return time.time() * time_scale


def get_time_scale():
    return time_scale


def find_nearest_index(array, value):
    """
    Find the index of the element nearest to value in an array.
    https://stackoverflow.com/a/26026189
    """
    idx = np.searchsorted(array, value, side="left")
    is_left = idx > 0 and (idx == len(array) or np.abs(
        value - array[idx-1]) < np.abs(value - array[idx]))
    return idx - is_left


def find_k_nearest_indices(array, value, k=1):
    """
    Find k nearest indices in an array, sorted by proximity.
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
    return indices
