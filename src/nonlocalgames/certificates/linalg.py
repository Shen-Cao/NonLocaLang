"""Exact rational positive-semidefiniteness verification without numerical algebra."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from numbers import Integral

from ..core.game import rational


@dataclass(frozen=True)
class PSDVerification:
    """Exact Schur-complement pivots and the resulting matrix rank."""
    rank: int
    pivots: tuple[Fraction, ...]


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def verify_psd(matrix, *, max_size=128):
    """Verify a real symmetric rational matrix by exact Schur elimination.

    A negative pivot rejects the matrix. A zero pivot requires its entire
    remaining row to vanish: otherwise a 2x2 principal minor is negative.
    A positive pivot permits an exact Schur complement. This criterion is
    necessary and sufficient, including singular matrices. The empty matrix
    is accepted with rank zero. No tolerance or eigensolver is used.
    """
    budget = positive_integer(max_size, "max_size")
    rows = tuple(tuple(islice(iter(row), budget + 1))
                 for row in islice(iter(matrix), budget + 1))
    size = len(rows)
    if size > budget or any(len(row) > budget for row in rows):
        raise ValueError("Exact PSD verification exceeds max_size")
    if any(len(row) != size for row in rows):
        raise ValueError("PSD verification requires a square matrix")
    residual = [list(map(rational, row)) for row in rows]
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
