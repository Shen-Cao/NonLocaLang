from copy import deepcopy
from fractions import Fraction
import json

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.certificates.xor_primal import verify_xor_gram, verify_xor_primal_certificate
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.discovery.xor3 import xor3_game
from nonlocalgames.games import biased_chsh, chsh


def test_rank_two_gram_attains_mask_10_optimum_with_qubit_dimensions(monkeypatch):
    import cvxpy as cp
    monkeypatch.setattr(cp.Problem, 'solve', lambda *a, **kw: pytest.fail('pure verifier called solver'))
    half = Fraction(1, 2)
    block = ((1, half, half), (half, 1, -half), (half, -half, 1))
    gram = tuple(row+row for row in block)*2
    game = xor3_game(10)
    result = verify_xor_gram(game, gram)
    saved = json.loads(json.dumps(result.witness))
    replay = verify_xor_primal_certificate(game, saved)
    assert result == replay
    assert result.value == Fraction(5, 6)
    assert result.guarantee == 'certified_lower_bound'
    assert result.diagnostics['gram_rank'] == 2
    assert result.diagnostics['realization_dimensions'] == (2, 2)


def test_rank_one_correlations_and_signed_payoff_with_an_unused_question():
    scenario = Scenario((2, 1), (2, 2))
    game = Game.from_rule(scenario, lambda x, a: 5*a[0] if x[0] else (2 if a[0] == a[1] else -3),
                          distribution={(0, 0): 1})
    sign = (1, 1, -1)
    result = verify_xor_gram(game, tuple(tuple(s*t for t in sign) for s in sign))
    assert result.value == -3 and result.diagnostics['realization_dimensions'] == (1, 1)
    with pytest.raises(ValueError, match='parity'):
        verify_xor_gram(Game(scenario, ('1/2', '1/2'), game.payoff), [[1]*3]*3)


def test_dimension_of_odd_rank_clifford_realization():
    game = Game.from_rule(Scenario((1, 2), (2, 2)), lambda x, a: a[0] == a[1])
    result = verify_xor_gram(game, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert result.value == Fraction(1, 2)
    assert result.diagnostics['gram_rank'] == 3
    assert result.diagnostics['realization_dimensions'] == (2, 2)


def test_invalid_gram_cannot_certify_an_algebraically_perfect_chsh_score():
    gram = ((1, 1, 1, 1), (1, 1, 1, -1), (1, 1, 1, 1), (1, -1, 1, 1))
    with pytest.raises(ValueError, match='not PSD'):
        verify_xor_gram(chsh(), gram)


def test_saved_gram_binds_score_and_full_game():
    payload = verify_xor_gram(chsh(), [[1]*4]*4).witness
    with pytest.raises(ValueError, match='different verifier'):
        verify_xor_primal_certificate(biased_chsh('1/10'), payload)
    with pytest.raises(ValueError, match='Claimed'):
        verify_xor_primal_certificate(chsh(), {**payload, 'achieved_score': '1'})
    with pytest.raises(ValueError, match='schema'):
        verify_xor_primal_certificate(chsh(), {**payload, 'rank': 1})


def test_saved_gram_requires_canonical_rational_entries_and_bound():
    payload = verify_xor_gram(chsh(), [[1]*4]*4).witness
    for invalid in (1, True, '1.0', '2/2'):
        damaged = deepcopy(payload)
        damaged['gram'][0][0] = invalid
        with pytest.raises(ValueError):
            verify_xor_primal_certificate(chsh(), damaged)
    with pytest.raises(ValueError):
        verify_xor_primal_certificate(chsh(), {**payload, 'achieved_score': '0.75'})
    damaged = deepcopy(payload)
    damaged['gram'][0][0] = '0'
    with pytest.raises(ValueError, match='diagonal'):
        verify_xor_primal_certificate(chsh(), damaged)
    damaged = deepcopy(payload)
    damaged['gram'][0][1] = '-1'
    with pytest.raises(ValueError, match='symmetry'):
        verify_xor_primal_certificate(chsh(), damaged)


def test_gram_shape_checked_before_raw_scalar_conversion(monkeypatch):
    import nonlocalgames.certificates.xor_primal as checker
    payload = verify_xor_gram(chsh(), [[1]*4]*4).witness
    monkeypatch.setattr(checker, 'parse_rational', lambda *a, **k: pytest.fail('shape must precede parse'))
    with pytest.raises(ValueError, match='size'):
        verify_xor_primal_certificate(chsh(), {**payload, 'gram': payload['gram'][:-1]})
    with pytest.raises(ValueError, match='size'):
        verify_xor_gram(chsh(), [[1]*3]*3)
    with pytest.raises(ValueError, match='limit'):
        verify_xor_gram(chsh(), object(), max_matrix_size=3)


def test_supplied_scalar_budgets_and_derived_objective_have_different_costs():
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: 5 if a[0] == a[1] else 0)
    point = verify_xor_gram(game, ((1, '1/3'), ('1/3', 1)), max_scalar_bits=3)
    assert point.value == Fraction(10, 3)
    with pytest.raises(ScalarBudgetExceeded):
        verify_xor_primal_certificate(game, point.witness, max_scalar_bits=3)
    assert verify_xor_primal_certificate(game, point.witness, max_scalar_bits=4).value == point.value
    with pytest.raises(ScalarBudgetExceeded):
        verify_xor_gram(game, ((1, '1e-1000000000'), ('1e-1000000000', 1)))
