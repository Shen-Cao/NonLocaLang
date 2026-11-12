"""Exact kernels must never inherit fixed-width arithmetic from scalar containers."""
from fractions import Fraction
import warnings

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import classical_value
from nonlocalgames.certificates.linalg import verify_psd
from nonlocalgames.core.game import rational


def bounded_fraction(numerator, denominator=1):
    return Fraction(np.int64(numerator), np.int64(denominator))


@pytest.mark.parametrize('value', [np.int64(2**62), np.int64(-(2**63)), np.uint64(2**64-1),
                                   bounded_fraction(1, 2**40)])
def test_rational_conversion_normalizes_to_unbounded_python_integers(value):
    exact = rational(value)
    assert type(exact.numerator) is int and type(exact.denominator) is int
    assert exact * exact == Fraction(int(exact.numerator)**2, int(exact.denominator)**2)


def test_game_coefficients_and_local_values_do_not_overflow_int64():
    probability = bounded_fraction(1, 2**40)
    payoff = bounded_fraction(1, 2**40-1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        game = Game(Scenario((2,), (1,)), (probability, Fraction(2**40-1, 2**40)), (payoff, 0))
        expected = Fraction(1, 2**40*(2**40-1))
        assert game.coefficient((0,), (0,)) == expected
        assert classical_value(game).value == expected


def test_exact_psd_checker_rejects_indefinite_matrix_even_if_numpy_product_would_wrap():
    # Determinant is 2**62 - 2**80 < 0; the off-diagonal square must not wrap to zero.
    matrix = ((bounded_fraction(2**62), bounded_fraction(2**40)),
              (bounded_fraction(2**40), bounded_fraction(1)))
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(ValueError, match='not PSD'):
            verify_psd(matrix)
    rank_one = ((bounded_fraction(2**62), bounded_fraction(2**40)),
                (bounded_fraction(2**40), bounded_fraction(2**18)))
    assert verify_psd(rank_one).rank == 1
