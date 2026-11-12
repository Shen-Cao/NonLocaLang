"""Bounded decoding of explicitly numerical complex matrices."""
from math import isfinite
from numbers import Integral, Real


def positive_budget(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def positive_tolerance(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError("tolerance must be a positive finite real number")
    return float(value)


def matrix_from_dict(data, dimension):
    """Check shape and real JSON-number entries before allocating NumPy matrices.

    Caller checks aggregate dimension/entry budgets before invoking this helper.
    Numeric strings and booleans are not valid complex_float64 entries.
    """
    if not isinstance(data, dict) or set(data) != {"real", "imag"}:
        raise ValueError("Numerical matrices require exactly real and imag fields")
    for part in (data["real"], data["imag"]):
        if (not isinstance(part, list) or len(part) != dimension
                or any(not isinstance(row, list) or len(row) != dimension for row in part)):
            raise ValueError("Matrix data does not match the declared square dimension")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for row in part for v in row):
            raise ValueError("Numerical matrix entries must be real JSON numbers")
        try:
            finite = all(isfinite(v) for row in part for v in row)
        except OverflowError as error:
            raise ValueError("Matrix entries exceed floating-point range") from error
        if not finite:
            raise ValueError("Numerical matrices must contain only finite numbers")
    import numpy as np
    matrix = np.empty((dimension, dimension), dtype=complex)
    matrix.real = np.asarray(data["real"], dtype=float)
    matrix.imag = np.asarray(data["imag"], dtype=float)
    return matrix
