"""Immutable complex rational matrices and exact positivity verification."""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import islice
from math import isqrt

from ...certificates.linalg import verify_psd
from ...core.scalar_wire import ScalarBudgetExceeded
from .budgets import ExactBudgetExceeded, coerce_budget
from .scalars import exact_scalar, scalar_bits


class ExactPSDViolation(ValueError):
    """A Hermitian rational matrix fails an exact positivity condition."""


def _rows(data, size_limit):
    rows = tuple(islice(iter(data), size_limit+1))
    if len(rows) > size_limit:
        raise ExactBudgetExceeded('Matrix exceeds dimension or entry budget')
    if not rows:
        raise ValueError('Matrices must be nonempty')
    return rows


def _grid(rows, size, max_bits):
    result = []
    for row in rows:
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeError('A matrix row must be an iterable of exact scalar entries')
        entries = tuple(islice(iter(row), size+1))
        if len(entries) != size:
            raise ValueError('Matrix data must be square with matching real/imaginary shapes')
        result.append(tuple(exact_scalar(v, max_bits=max_bits) for v in entries))
    return tuple(result)


@dataclass(frozen=True, init=False)
class RationalMatrix:
    """A square matrix over Q+iQ; no numerical conversion is implicit.

    General matrices are permitted for algebra. State/effect verification checks
    exact Hermiticity and PSD explicitly. Budget metadata is not part of equality.
    """
    real: tuple[tuple[Fraction, ...], ...]
    imag: tuple[tuple[Fraction, ...], ...]
    _budget: object = field(repr=False, compare=False)
    _max_bits: int = field(repr=False, compare=False)

    def __init__(self, real, imag=None, *, budget=None):
        budget = coerce_budget(budget)
        row_limit = min(budget.max_joint_dimension, isqrt(budget.max_matrix_entries))
        rows = _rows(real, row_limit)
        size = len(rows)
        if size*size > budget.max_matrix_entries:
            raise ExactBudgetExceeded('Matrix exceeds max_matrix_entries')
        real = _grid(rows, size, budget.max_scalar_bits)
        if imag is None:
            imag = ((Fraction(0),)*size,)*size
        else:
            rows = _rows(imag, row_limit)
            if len(rows) != size:
                raise ValueError('Real and imaginary parts must have identical dimensions')
            imag = _grid(rows, size, budget.max_scalar_bits)
        object.__setattr__(self, 'real', real)
        object.__setattr__(self, 'imag', imag)
        object.__setattr__(self, '_budget', budget)
        object.__setattr__(self, '_max_bits', max(scalar_bits(v) for grid in (real, imag) for row in grid for v in row))

    @property
    def dimension(self):
        return len(self.real)

    def check_budget(self, budget=None):
        budget = self._budget if budget is None else coerce_budget(budget)
        if self.dimension > budget.max_joint_dimension:
            raise ExactBudgetExceeded('Matrix exceeds max_joint_dimension')
        if self.dimension**2 > budget.max_matrix_entries:
            raise ExactBudgetExceeded('Matrix exceeds max_matrix_entries')
        if self._max_bits > budget.max_scalar_bits:
            raise ExactBudgetExceeded('Exact matrix entries exceed max_scalar_bits')
        return budget

    def trace(self, *, budget=None):
        budget = self.check_budget(budget)
        return tuple(exact_scalar(sum((grid[i][i] for i in range(self.dimension)), Fraction(0)),
                                  max_bits=budget.max_scalar_bits) for grid in (self.real, self.imag))

    def is_hermitian(self):
        return all(self.real[i][j] == self.real[j][i] and self.imag[i][j] == -self.imag[j][i]
                   for i in range(self.dimension) for j in range(self.dimension))


def realification(matrix, *, budget=None):
    """Return [[Re,-Im],[Im,Re]] only for checking; physical dimensions do not change."""
    if not isinstance(matrix, RationalMatrix):
        raise TypeError('Expected RationalMatrix')
    budget = matrix.check_budget(budget)
    if 2*matrix.dimension > budget.max_psd_size:
        raise ExactBudgetExceeded('Realification exceeds max_psd_size')
    return tuple(matrix.real[i] + tuple(-v for v in matrix.imag[i]) for i in range(matrix.dimension)) + tuple(
        matrix.imag[i] + matrix.real[i] for i in range(matrix.dimension))


def verify_psd_complex(matrix, *, budget=None):
    """Check complex PSD by exact Hermiticity and rational realification."""
    if not isinstance(matrix, RationalMatrix):
        raise TypeError('Expected RationalMatrix')
    budget = matrix.check_budget(budget)
    if not matrix.is_hermitian():
        raise ValueError('Matrix is not exactly Hermitian')
    converted = realification(matrix, budget=budget)
    try:
        return verify_psd(converted, max_size=budget.max_psd_size, max_scalar_bits=budget.max_scalar_bits)
    except ScalarBudgetExceeded as error:
        raise ExactBudgetExceeded('Complex PSD scalar verification exceeds the exact budget') from error
    except ValueError as error:
        if 'not PSD' in str(error):
            raise ExactPSDViolation(str(error)) from error
        raise
