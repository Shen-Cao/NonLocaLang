from fractions import Fraction
from math import sqrt

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import classical_value, npa_bound, nonsignalling_value, xor_quantum_value
from nonlocalgames.analysis.npa import reduce_word
from nonlocalgames.games import biased_chsh, chsh, ghz


@pytest.mark.parametrize("level", ["1", "1+AB"])
def test_chsh_npa(level):
    result = npa_bound(chsh(), level=level, solver="SCS", eps=1e-7)
    assert result.value == pytest.approx((2 + sqrt(2))/4, abs=3e-6)
    assert result.guarantee == "numerical_relaxation"
    assert not result.diagnostics["certified"]
    assert result.diagnostics["minimum_eigenvalue"] > -3e-6
    assert result.diagnostics["max_constraint_violation"] < 3e-6
    assert np.allclose(result.witness["behavior"].sum(axis=1), 1, atol=3e-6)


@pytest.mark.parametrize("p", [Fraction(0), Fraction(1, 4), Fraction(1)])
def test_xor_against_independent_npa_formulation(p):
    game = biased_chsh(p)
    xor = xor_quantum_value(game, solver="SCS", eps=1e-7)
    npa = npa_bound(game, solver="SCS", eps=1e-7)
    assert xor.value == pytest.approx(npa.value, abs=3e-6)
    assert xor.guarantee == "numerical_optimum"
    if p in (0, 1):
        assert xor.value == pytest.approx(1, abs=3e-6)


def test_nonbinary_npa_and_signed_rewards():
    game = Game.from_rule(Scenario((1, 2), (3, 2)),
                          lambda x, a: Fraction(a[0] - 2*a[1] + x[1], 3) - 2)
    for level in ("1", "1+AB"):
        result = npa_bound(game, level=level, solver="SCS", eps=1e-7)
        assert result.value == pytest.approx(float(classical_value(game).value), abs=3e-6)
        assert np.allclose(result.witness["behavior"].sum(axis=1), 1, atol=3e-6)


def test_npa_nested_relaxations_on_asymmetric_game():
    game = Game.from_rule(Scenario((2, 2), (2, 3)),
                          lambda x, a: Fraction((3*x[0] + x[1]*a[0] + 2*a[1]) % 7, 6))
    local = float(classical_value(game).value)
    q1 = npa_bound(game, level="1", solver="SCS", eps=1e-7).value
    qab = npa_bound(game, level="1+AB", solver="SCS", eps=1e-7).value
    ns = nonsignalling_value(game).value
    assert local <= qab + 3e-6 <= q1 + 6e-6 <= ns + 9e-6


def test_i3322_intermediate_level_published_regression():
    # The 1+AB value agrees with Table 3 of https://arxiv.org/abs/0803.4290.
    coefficients = ((1, 1, 1), (1, 1, -1), (1, -1, 0))
    game = Game.from_rule(
        Scenario((3, 3), (2, 2)),
        lambda x, a: 9*coefficients[x[0]][x[1]]*(a == (0, 0))
        - 3*(x[0] == 0)*(a[0] == 0)
        - 3*(2*(x[1] == 0) + (x[1] == 1))*(a[1] == 0),
    )
    assert classical_value(game).value == 0
    result = npa_bound(game, level="1+AB", solver="SCS", eps=1e-7)
    assert result.value == pytest.approx(0.2515, abs=5e-5)


def test_projector_rewriting_does_not_commute_same_party():
    a0, a1, a0other, b = (0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)
    assert reduce_word((a0, b, a0)) == (a0, b)
    assert reduce_word((a0, b, a0other)) is None
    assert reduce_word((a0, a1, a0)) == (a0, a1, a0)


def test_backend_scope_and_size_guards():
    with pytest.raises(ValueError, match="two parties"):
        npa_bound(ghz())
    with pytest.raises(ValueError, match="supported NPA levels"):
        npa_bound(chsh(), level="2")
    with pytest.raises(ValueError, match="exceeds limit"):
        npa_bound(chsh(), max_matrix_size=2)
    with pytest.raises(ValueError, match="exceeds limit"):
        xor_quantum_value(chsh(), max_matrix_size=2)
    nonxor = Game.from_rule(Scenario((2, 2), (2, 2)), lambda x, a: a[0])
    with pytest.raises(ValueError, match="XOR"):
        xor_quantum_value(nonxor)


def test_inaccurate_solver_status_is_retained():
    with pytest.warns(UserWarning, match="inaccurate"):
        result = xor_quantum_value(chsh(), solver="SCS", max_iters=1)
    assert result.status == "optimal_inaccurate"
    assert result.diagnostics["inaccurate"]
    assert not result.diagnostics["certified"]
