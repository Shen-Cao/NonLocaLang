"""Exact complex matrix algebra in the original physical tensor dimensions."""
from fractions import Fraction

from .budgets import ExactBudgetExceeded, bounded_dimension, coerce_budget
from .matrix import RationalMatrix
from .scalars import exact_scalar


def _pair(left, right, budget):
    if not isinstance(left, RationalMatrix) or not isinstance(right, RationalMatrix):
        raise TypeError('Exact algebra requires RationalMatrix operands')
    budget = left.check_budget(budget)
    right.check_budget(budget)
    return budget


def identity(dimension, *, budget=None):
    budget = coerce_budget(budget)
    dimension = bounded_dimension(dimension, budget)
    if dimension*dimension > budget.max_matrix_entries:
        raise ExactBudgetExceeded('Identity exceeds max_matrix_entries')
    return RationalMatrix(tuple(tuple(int(i == j) for j in range(dimension)) for i in range(dimension)), budget=budget)


def add(left, right, *, budget=None):
    budget = _pair(left, right, budget)
    if left.dimension != right.dimension:
        raise ValueError('Matrix dimensions differ')
    grids = [tuple(tuple(a+b for a, b in zip(lr, rr)) for lr, rr in zip(lg, rg))
             for lg, rg in ((left.real, right.real), (left.imag, right.imag))]
    return RationalMatrix(*grids, budget=budget)


def scale(matrix, scalar, *, budget=None):
    if not isinstance(matrix, RationalMatrix):
        raise TypeError('Expected RationalMatrix')
    budget = matrix.check_budget(budget)
    scalar = exact_scalar(scalar, max_bits=budget.max_scalar_bits)
    return RationalMatrix(tuple(tuple(scalar*v for v in row) for row in matrix.real),
                          tuple(tuple(scalar*v for v in row) for row in matrix.imag), budget=budget)


def subtract(left, right, *, budget=None):
    budget = _pair(left, right, budget)
    return add(left, scale(right, -1, budget=budget), budget=budget)


def adjoint(matrix, *, budget=None):
    if not isinstance(matrix, RationalMatrix):
        raise TypeError('Expected RationalMatrix')
    budget = matrix.check_budget(budget)
    return RationalMatrix(tuple(zip(*matrix.real)),
                          tuple(tuple(-v for v in row) for row in zip(*matrix.imag)), budget=budget)


def kron(left, right, *, budget=None):
    budget = _pair(left, right, budget)
    size = left.dimension*right.dimension
    if size > budget.max_joint_dimension or size*size > budget.max_matrix_entries:
        raise ExactBudgetExceeded('Tensor product exceeds exact matrix budgets')
    real, imag = [], []
    for i in range(left.dimension):
        for k in range(right.dimension):
            real.append(tuple(left.real[i][j]*right.real[k][column] - left.imag[i][j]*right.imag[k][column]
                              for j in range(left.dimension) for column in range(right.dimension)))
            imag.append(tuple(left.real[i][j]*right.imag[k][column] + left.imag[i][j]*right.real[k][column]
                              for j in range(left.dimension) for column in range(right.dimension)))
    return RationalMatrix(real, imag, budget=budget)


def multiply(left, right, *, budget=None):
    budget = _pair(left, right, budget)
    if left.dimension != right.dimension:
        raise ValueError('Matrix dimensions differ')
    n = left.dimension
    real = tuple(tuple(sum((left.real[i][k]*right.real[k][j] - left.imag[i][k]*right.imag[k][j]
                            for k in range(n)), Fraction(0)) for j in range(n)) for i in range(n))
    imag = tuple(tuple(sum((left.real[i][k]*right.imag[k][j] + left.imag[i][k]*right.real[k][j]
                            for k in range(n)), Fraction(0)) for j in range(n)) for i in range(n))
    return RationalMatrix(real, imag, budget=budget)


def trace_product(left, right, *, budget=None):
    """Compute tr(left*right) without materializing their matrix product."""
    budget = _pair(left, right, budget)
    if left.dimension != right.dimension:
        raise ValueError('Matrix dimensions differ')
    real, imag = Fraction(0), Fraction(0)
    for i in range(left.dimension):
        for j in range(left.dimension):
            real = exact_scalar(real + left.real[i][j]*right.real[j][i] - left.imag[i][j]*right.imag[j][i],
                                max_bits=budget.max_scalar_bits)
            imag = exact_scalar(imag + left.real[i][j]*right.imag[j][i] + left.imag[i][j]*right.real[j][i],
                                max_bits=budget.max_scalar_bits)
    return real, imag
