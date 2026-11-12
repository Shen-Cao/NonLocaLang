"""Exact finite witnesses establish achieved scores with full normalization."""
from fractions import Fraction

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.games import chsh, ghz
from nonlocalgames.quantum.exact import (ExactBudget, ExactBudgetExceeded, ExactPSDViolation,
                                         ExactQuantumStrategy, RationalMatrix,
                                         exact_chsh_strategy, exact_ghz_strategy,
                                         validate_game_scalars)
from nonlocalgames.quantum.exact.algebra import identity, kron, scale


def test_exact_chsh_and_complex_ghz_scores():
    chsh_witness = exact_chsh_strategy(Fraction(2, 5))
    assert chsh_witness.score(chsh()) == Fraction(99, 116)
    assert exact_chsh_strategy().score(chsh()) == Fraction(3880899, 4546756)
    ghz_witness = exact_ghz_strategy()
    assert ghz_witness.dimensions == (2, 2, 2)
    assert ghz_witness.score(ghz()) == 1
    assert ghz_witness.validate()['state_rank'] == 1
    probabilities = ghz_witness.probabilities(ghz().scenario)
    for row in range(8):
        assert sum(probabilities[8*row:8*row+8]) == 1
    assert all(isinstance(p, Fraction) and 0 <= p <= 1 for p in probabilities)


def test_signed_nonbinary_unequal_tensor_factors():
    plus_i = RationalMatrix(((Fraction(1, 2), 0), (0, Fraction(1, 2))),
                             ((0, Fraction(-1, 2)), (Fraction(1, 2), 0)))
    second = RationalMatrix(((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    zero = scale(identity(2), 0)
    bob = tuple(RationalMatrix(tuple(tuple(int(i == j == a) for j in range(3)) for i in range(3)))
                for a in range(3))
    witness = ExactQuantumStrategy(kron(plus_i, second), (((zero, identity(2)),), (bob,)), (2, 3))
    game = Game.from_rule(Scenario((1, 1), (2, 3)), lambda x, a: Fraction(-7, 3) if a == (1, 1) else 4)
    assert witness.score(game) == Fraction(-7, 3)
    assert witness.probabilities()[4] == 1
    assert witness.dimensions == (2, 3)


def test_exact_rejects_bad_state_and_povm_including_unused_input():
    unit = identity(1)
    with pytest.raises(ValueError, match='trace'):
        ExactQuantumStrategy(scale(unit, 2), (((unit,),),))
    bad = RationalMatrix(((1, 0), (0, -1)))
    with pytest.raises(ExactPSDViolation):
        ExactQuantumStrategy(scale(identity(2), Fraction(1, 2)), (((bad, identity(2)),),))
    with pytest.raises(ValueError, match='sum to identity'):
        ExactQuantumStrategy(unit, (((unit,), (scale(unit, 0),)),))
    with pytest.raises(ValueError, match='tensor factors'):
        ExactQuantumStrategy(scale(identity(2), Fraction(1, 2)), (((unit,),),))
    witness = ExactQuantumStrategy(unit, (((unit,),),))
    promised = Game.from_rule(Scenario((2,), (1,)), lambda x, a: 1, distribution={(0,): 1})
    with pytest.raises(ValueError, match='interfaces'):
        witness.score(promised)


def test_event_party_and_aggregate_matrix_guards():
    witness = exact_ghz_strategy()
    with pytest.raises(ExactBudgetExceeded, match='max_events'):
        witness.score(ghz(), budget=ExactBudget(max_events=63))
    with pytest.raises(ExactBudgetExceeded, match='max_parties'):
        witness.validate(budget=ExactBudget(max_parties=2))
    with pytest.raises(ExactBudgetExceeded, match='max_matrix_entries'):
        exact_chsh_strategy(budget=ExactBudget(max_matrix_entries=20))
    with pytest.raises(ExactBudgetExceeded, match='max_scalar_bits'):
        witness.score(Game.from_rule(ghz().scenario, lambda x, a: 1 << 4096))


def test_raw_cancelled_or_unused_game_data_is_checked_before_hashing():
    huge = 1 << 4096
    scenario = Scenario((2,), (1,))
    game = Game(scenario, (1, 0), (1, huge))
    with pytest.raises(ExactBudgetExceeded, match='max_scalar_bits'):
        validate_game_scalars(game)
    game = Game(scenario, (Fraction(1, huge), 1-Fraction(1, huge)), (huge, 0))
    with pytest.raises(ExactBudgetExceeded, match='max_scalar_bits'):
        validate_game_scalars(game)


def test_exact_checks_need_no_numerical_linear_algebra(monkeypatch):
    import numpy as np
    import cvxpy as cp
    def forbidden(*args, **kwargs):
        pytest.fail('exact verification must not call numerical algebra or a solver')
    monkeypatch.setattr(np.linalg, 'eigvalsh', forbidden)
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    monkeypatch.setattr(cp.Problem, 'solve', forbidden)
    assert exact_ghz_strategy().score(ghz()) == 1


def test_verifier_coefficients_use_unbounded_python_integer_arithmetic():
    import numpy as np
    first = Fraction(np.int64(1), np.int64(2**40))
    reward = Fraction(np.int64(1), np.int64(2**40-1))
    game = Game(Scenario((2,), (1,)), (first, 1-first), (reward, 0))
    coefficients = validate_game_scalars(game)
    assert coefficients == (Fraction(1, 2**40*(2**40-1)), Fraction(0))
