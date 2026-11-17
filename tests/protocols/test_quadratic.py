from fractions import Fraction as F
from itertools import repeat

import numpy as np
import pytest

from nonlocalgames import Scenario
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.networks.behavior import RationalBehavior
from nonlocalgames.protocols.quadratic import Quadratic as Q, scalar_from_pair, scalar_to_pair
from nonlocalgames.protocols.quadratic_behavior import QuadraticBehavior
from nonlocalgames.protocols.wiring import WiringBudgetExceeded


def test_exact_arithmetic_and_lossless_rational_projection():
    x = Q(F(-7, 3), F(5, 3))
    assert x * Q(-7, -5) == F(-1, 3)
    assert (x + 2) - Q(2) == -(-x)
    assert 1 - Q(1, 1) == Q(0, -1)
    assert (3*x/3) == x
    assert (Q(0, 1)*Q(0, 1)).as_fraction() == 2
    assert not Q(0) and Q(0, 1)
    with pytest.raises(ValueError, match='rational projection'):
        x.as_fraction()
    with pytest.raises(ZeroDivisionError):
        x/0
    assert hash(Q(1)) == hash(1)
    for value in (True, 0.5, '1/2', 1+0j):
        with pytest.raises(TypeError):
            Q(value)
    foreign = Q(F(np.int64(5), np.int64(7)), np.int64(8))
    assert type(foreign.a.numerator) is int and type(foreign.b.numerator) is int


def test_sign_resolves_both_sides_of_close_pell_approximations():
    a, b = 1, 0
    for k in range(1, 129):
        a, b = a+2*b, a+b
        expected = (-1)**k
        assert a*a - 2*b*b == expected
        assert Q(a, -b).sign() == expected
        assert Q(-a, b).sign() == -expected
    assert Q(0, -1).sign() == -1 and Q(0).sign() == 0


def test_wire_uses_unique_pairs_of_bounded_canonical_rational_text():
    q = Q(F(2, 7), F(-1, 5))
    assert scalar_from_pair(scalar_to_pair(q)) == q
    for malformed in ({"a": "1", "b": "0"}, ["2/4", "0"], ["1", "0.0"], [0, "1"], ["1"]):
        with pytest.raises(ValueError):
            scalar_from_pair(malformed)
    with pytest.raises(ScalarBudgetExceeded):
        scalar_from_pair(["1/8", "0"], max_scalar_bits=3)
    with pytest.raises(ScalarBudgetExceeded):
        scalar_from_pair(["1", "-1/4"], max_scalar_chars=2)


def test_positivity_and_normalization_use_both_exact_coefficients():
    s = Scenario((1,), (2,))
    p = Q(99, -70)
    valid = QuadraticBehavior(s, (p, 1-p))
    assert valid.marginal((), (0,)) == {(): Q(1)}
    for bad in ((-p, 1+p), (Q(239, -169), Q(-238, 169))):
        with pytest.raises(ValueError, match='nonnegative'):
            QuadraticBehavior(s, bad)
    with pytest.raises(ValueError, match='sum exactly'):
        QuadraticBehavior(s, (p, 1-p+Q(0, F(1, 1024))))


def test_complement_nonsignalling_detects_changes_hidden_in_correlations():
    # Only the joint (first, last) correlations depend on the middle question.
    s = Scenario((1, 2, 1), (2, 1, 2))
    p = Q(F(1, 4), F(1, 16))
    q = Q(F(1, 2)) - p
    with pytest.raises(ValueError, match='signals from party 1'):
        QuadraticBehavior(s, (p, q, q, p, q, p, p, q))
    valid = QuadraticBehavior(s, (p, q, q, p)*2)
    assert valid.marginal((2, 0), (0, 1, 0)) == {(0, 0): p, (1, 0): q, (0, 1): q, (1, 1): p}
    with pytest.raises(ValueError):
        valid.marginal(repeat(1), (0, 0, 0))


def test_budgets_cover_counts_and_coefficient_growth_during_validation():
    def untouched():
        pytest.fail("shape budget must precede probability reads")
        yield
    with pytest.raises(WiringBudgetExceeded, match='max_entries'):
        QuadraticBehavior(Scenario((2**100,), (3,)), untouched())
    with pytest.raises(WiringBudgetExceeded, match='max_parties'):
        QuadraticBehavior(Scenario((1, 1), (1, 1)), untouched(), max_parties=1)
    with pytest.raises(ValueError, match='length'):
        QuadraticBehavior(Scenario((1,), (2,)), repeat(Q(0)))
    s = Scenario((1,), (5,))
    p = (Q(F(1, 6)), Q(F(1, 4)), Q(F(1, 6)), Q(F(1, 4)), Q(F(1, 6)))
    with pytest.raises(ScalarBudgetExceeded):
        QuadraticBehavior(s, p, max_scalar_bits=3)
    assert QuadraticBehavior(s, p, max_scalar_bits=4).as_rational() == RationalBehavior(s, [v.a for v in p])
