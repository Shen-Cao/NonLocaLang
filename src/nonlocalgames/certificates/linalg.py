"""Exact rational positive-semidefiniteness verification without numerical algebra."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from numbers import Integral

from ..core.scalar_wire import parse_rational


@dataclass(frozen=True)
class PSDVerification:
    """Exact Schur-complement pivots and the resulting matrix rank."""
    rank: int
    pivots: tuple[Fraction, ...]


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bounded_rows(matrix, max_size):
    budget = positive_integer(max_size, "max_size")
    if isinstance(matrix, (str, bytes, bytearray)):
        raise TypeError("PSD verification requires an iterable of matrix rows")
    outer = tuple(islice(iter(matrix), budget + 1))
    if len(outer) > budget:
        raise ValueError("Exact PSD verification exceeds max_size")
    rows = []
    for row in outer:
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeError("PSD matrix rows must be iterables of scalar entries")
        entries = tuple(islice(iter(row), budget + 1))
        if len(entries) > budget:
            raise ValueError("Exact PSD verification exceeds max_size")
        rows.append(entries)
    if any(len(row) != len(rows) for row in rows):
        raise ValueError("PSD verification requires a square matrix")
    return tuple(rows)


def verify_psd(matrix, *, max_size=128, max_scalar_bits=4096, max_scalar_chars=8192):
    """Verify a real symmetric rational matrix through bounded raw-data parsing.

    Text decoding and supplied integer components obey the scalar budgets before
    arithmetic. Exact Schur intermediates may grow; no wall-clock bound is claimed.
    Singular and empty matrices are supported without numerical tolerances.
    """
    bits = positive_integer(max_scalar_bits, 'max_scalar_bits')
    chars = positive_integer(max_scalar_chars, 'max_scalar_chars')
    rows = _bounded_rows(matrix, max_size)
    prepared = tuple(tuple(parse_rational(value, max_bits=bits, max_chars=chars) for value in row)
                     for row in rows)
    return _verify_psd_prepared(prepared, max_size=max_size)


def _verify_psd_prepared(matrix, *, max_size=128):
    """Check internally prepared native Fractions without imposing new wire limits.

    This kernel accepts no strings, floats, integers, or bounded-integer Fraction
    components. Callers must first validate raw data and perform their exact model
    arithmetic. Derived entries may exceed the source wire's bit budget.
    """
    rows = _bounded_rows(matrix, max_size)
    if any(type(value) is not Fraction or type(value.numerator) is not int
           or type(value.denominator) is not int for row in rows for value in row):
        raise TypeError('Prepared PSD entries must be native Fractions with builtin integer components')
    size = len(rows)
    residual = [list(row) for row in rows]
    if any(residual[i][j] != residual[j][i] for i in range(size) for j in range(i)):
        raise ValueError("PSD verification requires exact symmetry")
    pivots = []
    for k in range(size):
        pivot = residual[k][k]
        pivots.append(pivot)
        if pivot < 0:
            raise ValueError(f"Matrix is not PSD: negative exact pivot at index {k}")
        if pivot == 0:
            if any(residual[k][j] != 0 for j in range(k + 1, size)):
                raise ValueError(f"Matrix is not PSD: zero pivot with nonzero remaining row at index {k}")
            continue
        column = tuple(residual[i][k] for i in range(k + 1, size))
        for i in range(k + 1, size):
            for j in range(i, size):
                value = residual[i][j] - column[i-k-1] * column[j-k-1] / pivot
                residual[i][j] = residual[j][i] = value
    return PSDVerification(sum(p > 0 for p in pivots), tuple(pivots))
