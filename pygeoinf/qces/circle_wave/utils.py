import numpy as np


def is_1d(arr: np.ndarray) -> bool:
    """
    Checks an array is 1d in shape.
    """
    return arr.ndim == 1


def is_sorted(arr: np.ndarray) -> bool:
    """Checks if a 1D NumPy array is sorted."""
    return is_ascending(arr) or is_descending(arr)


def is_ascending(arr: np.ndarray) -> bool:
    """
    Returns true if the elements of a 1D array are in ascending order.
    """
    if not is_1d(arr):
        raise ValueError("Function applies only to 1D arrays")
    return np.all(arr[:-1] <= arr[1:])


def is_descending(arr: np.ndarray) -> bool:
    """
    Returns true if the elements of a 1D array are in descending order.
    """
    if not is_1d(arr):
        raise ValueError("Function applies only to 1D arrays")
    return np.all(arr[:-1] >= arr[1:])
