from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import nonsignalling_value
from nonlocalgames.games import chsh, ghz, gyni, magic_square


@pytest.mark.parametrize("constructor, expected", [(chsh, 1), (ghz, 1), (gyni, 1/3), (magic_square, 1)])
def test_benchmarks_and_all_marginals(constructor, expected):
    game = constructor()
    result = nonsignalling_value(game)
    assert result.value == pytest.approx(expected, abs=1e-8)
    assert result.guarantee == "numerical_optimum"
    assert result.diagnostics["max_equality_residual"] < 1e-8
    assert abs(result.diagnostics["primal_dual_gap"]) < 1e-8
    behavior = result.witness
    assert np.allclose(behavior.sum(axis=1), 1)
    assert behavior.min() >= -1e-8
    # Check each complement marginal for arbitrary pairs of inputs, independently
    # of the baseline-input construction used by the LP backend.
    for party in range(game.scenario.parties):
        for i, x in enumerate(game.inputs):
            for j, y in enumerate(game.inputs):
                if any(x[k] != y[k] for k in range(game.scenario.parties) if k != party):
                    continue
                for a in game.outputs:
                    indices = [k for k, b in enumerate(game.outputs)
                               if all(a[t] == b[t] for t in range(game.scenario.parties) if t != party)]
                    assert behavior[i, indices].sum() == pytest.approx(behavior[j, indices].sum(), abs=1e-8)


def test_nonbinary_no_input_and_signed_payoff():
    game = Game.from_rule(Scenario((1, 1), (3, 2)), lambda x, a: Fraction(a[0] - a[1] - 3, 2))
    assert nonsignalling_value(game).value == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="limit"):
        nonsignalling_value(chsh(), max_variables=1)


def test_singleton_input_parties_skip_empty_complement_enumeration(monkeypatch):
    import nonlocalgames.analysis.nonsignalling as nonsignalling
    game = Game(Scenario((1,)*64, (1,)*64), (1,), (1,))

    def forbidden(*args, **kwargs):
        pytest.fail('A singleton input cannot contribute a nonsignalling comparison row')

    monkeypatch.setattr(nonsignalling, 'product', forbidden)
    matrix, rhs = nonsignalling._nonsignalling_program(game, max_variables=1)
    assert matrix.shape == (1, 1) and matrix.nnz == 1
    assert matrix[0, 0] == 1 and rhs == (1,)


def test_mixed_input_sizes_keep_joint_complement_constraints_and_row_schema():
    from nonlocalgames.analysis.nonsignalling import _nonsignalling_program
    scenario = Scenario((1, 2, 1, 2), (2, 2, 2, 2))
    game = Game.from_rule(scenario, lambda x, a: 0)
    matrix, rhs = _nonsignalling_program(game, max_variables=64)
    assert matrix.shape == (36, 64)
    assert rhs == (1,)*4 + (0,)*32
    # Each singleton marginal is independent of remote questions, but the joint
    # marginal of the two inputless parties encodes party 1's question.
    signaling = np.array([.5 if a[1] == a[3] == 0 and (a[0] ^ a[2]) == x[1] else 0
                          for x in scenario.inputs for a in scenario.outputs])
    assert np.any(matrix @ signaling != np.asarray(rhs))
    independent = np.full(64, 1/16)
    np.testing.assert_array_equal(matrix @ independent, rhs)
