"""Physical feasibility checks and independent exact/numerical CHSH checks."""

from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.quantum import (
    QuantumStrategy, chsh_strategy, ghz_strategy, magic_square_strategy, verify_chsh_sos,
)
from nonlocalgames.games import ghz, magic_square


def chsh_game():
    scenario = Scenario((2, 2), (2, 2))
    payoff = tuple(Fraction((a[0] ^ a[1]) == x[0] * x[1]) for x in scenario.inputs for a in scenario.outputs)
    return Game(scenario, (Fraction(1, 4),) * 4, payoff, "CHSH")


def test_chsh_witness():
    game = chsh_game()
    strategy = chsh_strategy()
    probabilities = strategy.probabilities(game.scenario).reshape(4, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    assert probabilities.min() >= 0
    assert strategy.score(game) == pytest.approx(0.5 + np.sqrt(2) / 4, abs=1e-12)
    assert strategy.validate()["normalization_residual"] < 1e-12


def test_complex_conjugation_preserves_probabilities():
    vector = np.array([1, 1j, -1, 2j], dtype=complex) / np.sqrt(7)
    y = np.array([[0, -1j], [1j, 0]])
    x = np.array([[0, 1], [1, 0]])
    identity = np.eye(2)
    povms = tuple(tuple(((identity + o) / 2, (identity - o) / 2) for o in pair) for pair in ((y, x), (x, y)))
    strategy = QuantumStrategy(np.outer(vector, vector.conj()), povms)
    scenario = Scenario((2, 2), (2, 2))
    np.testing.assert_allclose(strategy.probabilities(scenario), strategy.conjugate().probabilities(scenario))
    assert not np.allclose(strategy.state, strategy.conjugate().state)


def test_multipartite_nonbinary_povms():
    state = np.eye(8) / 8
    one_povm = tuple(np.eye(2) / 3 for _ in range(3))
    strategy = QuantumStrategy(state, ((one_povm,),) * 3)
    scenario = Scenario((1, 1, 1), (3, 3, 3))
    np.testing.assert_allclose(strategy.probabilities(scenario), np.full(27, 1 / 27))


@pytest.mark.parametrize("state", [np.diag([1.1, -0.1]), np.eye(2), [[1, 1j], [0, 0]], [[np.nan, 0], [0, 1]]])
def test_invalid_states_rejected(state):
    with pytest.raises(ValueError):
        QuantumStrategy(state, (((np.eye(2),),),))


def test_invalid_povms_rejected():
    with pytest.raises(ValueError, match="normalized"):
        QuantumStrategy(np.eye(2) / 2, (((np.eye(2) / 2,),),))
    with pytest.raises(ValueError, match="positive"):
        QuantumStrategy(np.eye(2) / 2, (((np.diag([2, 0]), np.diag([-1, 1])),),))
    with pytest.raises(ValueError, match="dimension"):
        QuantumStrategy(np.eye(2) / 2, (((np.eye(3),),),))


def test_strategy_copies_inputs_and_rejects_alphabet_mismatch():
    state = np.eye(2) / 2
    strategy = QuantumStrategy(state, (((np.eye(2),),),))
    state[0, 0] = -1
    assert strategy.state[0, 0] == 0.5
    assert not strategy.state.flags.writeable
    with pytest.raises(ValueError, match="alphabets"):
        strategy.probabilities(Scenario((1,), (2,)))
    with pytest.raises(ValueError, match="party counts"):
        strategy.probabilities(Scenario((1, 1), (1, 1)))


def test_exact_sos_and_attainment():
    certificate = verify_chsh_sos()
    assert certificate.verified
    assert set(certificate.residuals.values()) == {"0"}
    assert certificate.bell_bound == "2*sqrt(2)"


@pytest.mark.parametrize("constructor, reference, dimensions", [
    (ghz, ghz_strategy, (2, 2, 2)),
    (magic_square, magic_square_strategy, (4, 4)),
])
def test_perfect_parity_and_consistency_references(constructor, reference, dimensions):
    game, strategy = constructor(), reference()
    assert strategy.dimensions == dimensions
    assert strategy.score(game) == pytest.approx(1, abs=1e-12)
    probabilities = strategy.probabilities(game.scenario).reshape(len(game.inputs), len(game.outputs))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    for i, x in enumerate(game.inputs):
        if game.probability(x):
            losing = [j for j, a in enumerate(game.outputs) if game.reward(x, a) == 0]
            assert probabilities[i, losing].sum() == pytest.approx(0, abs=1e-12)
    for party in strategy.measurements:
        for povm in party:
            for effect in povm:
                np.testing.assert_allclose(effect @ effect, effect, atol=1e-12)
