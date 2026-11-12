"""Exact complex algebra, bounded scalar parsing, and realification checks."""
from fractions import Fraction

import pytest

from nonlocalgames.quantum.exact import (ExactBudget, ExactBudgetExceeded, ExactPSDViolation,
                                         RationalMatrix, realification, verify_psd_complex)
from nonlocalgames.quantum.exact.algebra import (add, adjoint, identity, kron, multiply,
                                                scale, subtract, trace_product)
from nonlocalgames.quantum.exact.scalars import exact_scalar, scalar_string


def test_scalar_parser_accepts_only_explicit_bounded_exact_data():
    for value in (1, Fraction(1, 3), '-7/11', '0'):
        assert isinstance(exact_scalar(value), Fraction)
    for value in (True, 0.5, 1+0j):
        with pytest.raises(TypeError):
            exact_scalar(value)
    for text in ('1e1000000000', '0.5', '2/4', '+1', '-0', '1/1', '00', ' 1', '1/0'):
        with pytest.raises(ValueError):
            exact_scalar(text)
    for value in (1 << 4096, Fraction(1, 1 << 4096), '9'*2000):
        with pytest.raises(ExactBudgetExceeded):
            exact_scalar(value)
    value = Fraction((1 << 15000)+1, 3)
    encoded = scalar_string(value)
    assert exact_scalar(encoded, max_bits=16000) == value


def test_matrix_snapshots_are_immutable_and_general_complex_algebra_is_exact():
    real = [[1, 0], [0, 1]]
    unit = RationalMatrix(real)
    real[0][0] = 2
    assert unit.real[0][0] == 1
    with pytest.raises(TypeError):
        unit.real[0][0] = 2
    x = RationalMatrix(((0, 1), (1, 0)))
    y = RationalMatrix(((0, 0), (0, 0)), ((0, -1), (1, 0)))
    z = RationalMatrix(((1, 0), (0, -1)))
    assert multiply(y, y) == unit
    xy = multiply(x, y)
    assert xy.real == ((0, 0), (0, 0))
    assert xy.imag == z.real
    assert not xy.is_hermitian()
    assert add(xy, adjoint(xy)) == scale(unit, 0)
    assert subtract(x, x) == scale(unit, 0)
    assert trace_product(x, y) == (0, 0)
    assert trace_product(y, y) == (2, 0)


def test_realification_checks_psd_without_changing_physical_dimension():
    state = RationalMatrix(((Fraction(1, 2), 0), (0, Fraction(1, 2))),
                            ((0, Fraction(-1, 2)), (Fraction(1, 2), 0)))
    assert state.dimension == 2
    assert len(realification(state)) == 4
    assert verify_psd_complex(state).rank == 2
    assert state.trace() == (1, 0)
    with pytest.raises(ExactPSDViolation):
        verify_psd_complex(RationalMatrix(((1, 1), (1, 1-Fraction(1, 10**100)))))
    with pytest.raises(ValueError, match='Hermitian'):
        verify_psd_complex(RationalMatrix(((0, 0), (0, 0)), ((0, 1), (1, 0))))
    with pytest.raises(ExactBudgetExceeded):
        verify_psd_complex(state, budget=ExactBudget(max_psd_size=3))


def test_complex_tensor_product_has_correct_sign_and_original_dimension():
    y = RationalMatrix(((0, 0), (0, 0)), ((0, -1), (1, 0)))
    yy = kron(y, y)
    assert yy.dimension == 4
    assert yy.imag == ((0,)*4,)*4
    assert yy.real[0][3] == -1 and yy.real[1][2] == 1
    assert multiply(yy, yy) == identity(4)


def test_exact_budgets_precede_expansion_and_invalid_shapes_are_not_psd_failures():
    with pytest.raises(ExactBudgetExceeded):
        identity(10000, budget=ExactBudget(max_joint_dimension=10000, max_matrix_entries=1))
    with pytest.raises(ExactBudgetExceeded):
        kron(identity(4), identity(4), budget=ExactBudget(max_joint_dimension=8))
    def rows():
        yield (1,)
        yield (0,)
        pytest.fail('must stop once entry budget excludes the second row')
    with pytest.raises(ExactBudgetExceeded):
        RationalMatrix(rows(), budget=ExactBudget(max_joint_dimension=10000, max_matrix_entries=1))
    with pytest.raises(ValueError):
        RationalMatrix(((1, 0),))
    with pytest.raises(TypeError):
        RationalMatrix(['1'])


def test_fraction_components_are_normalized_out_of_numpy_integer_arithmetic():
    import numpy as np
    supplied = Fraction(np.int64(1), np.int64(2**40))
    scalar = exact_scalar(supplied)
    assert type(scalar.numerator) is int and type(scalar.denominator) is int
    left = RationalMatrix(((supplied,),))
    right = RationalMatrix(((Fraction(np.int64(1), np.int64(2**40-1)),),))
    assert multiply(left, right).real[0][0] == Fraction(1, 2**40*(2**40-1))


def test_fraction_like_subclasses_cannot_supply_truncated_components():
    class NonIntegralFraction(Fraction):
        @property
        def numerator(self):
            return 1.75
    with pytest.raises(TypeError):
        exact_scalar(NonIntegralFraction(1, 2))
