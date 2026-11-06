"""Returned deterministic witnesses can be rechecked under their exact resource contracts."""
from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import AnalysisTask, Game, ResourceSpec, Scenario
from nonlocalgames.classical import DeterministicStrategy, deterministic_quantum_strategy
from nonlocalgames.core.evaluation import evaluate
from nonlocalgames.games import chsh, ghz
from nonlocalgames.quantum import FixedState


@pytest.mark.parametrize('resource', [
    ResourceSpec('local'),
    ResourceSpec('local', shared_randomness='forbidden'),
    ResourceSpec('quantum_finite', (1, 2), state_constraint='product', measurement_class='projective'),
])
def test_optimizer_response_tables_have_exact_recheck(resource):
    task = AnalysisTask(chsh(), resource)
    result = task.solve()
    checked = evaluate(task, result.witness)
    assert checked.value == result.value == Fraction(3, 4)
    assert checked.guarantee == 'exact_witness'
    assert checked.status == 'evaluated'
    assert 'optimality is not asserted' in checked.diagnostics['scope']


def test_multiplayer_signed_score_matches_exact_behavior_contraction():
    scenario = Scenario((2, 1, 2), (3, 2, 1))
    game = Game.from_rule(scenario, lambda x, a: 2*a[0] - 3*a[1] + x[2])
    strategy = DeterministicStrategy(scenario, ((2, 1), (1,), (0, 0)))
    behavior = strategy.probabilities()
    assert strategy.score(game) == sum(c*p for c, p in zip(game.coefficients, behavior))
    assert all(isinstance(p, Fraction) for p in behavior)
    assert sum(behavior) == len(scenario.inputs)
    result = evaluate(AnalysisTask(game, ResourceSpec('local')), strategy)
    assert result.value == Fraction(1, 2)


def test_quantum_embedding_preserves_behavior_and_declared_product_restrictions():
    task = AnalysisTask(ghz(), ResourceSpec('quantum_finite', (1, 2, 3),
                                          state_constraint='product', measurement_class='projective'))
    tables = task.solve().witness
    classical = DeterministicStrategy(task.game.scenario, tables)
    quantum = deterministic_quantum_strategy(task.game.scenario, tables)
    assert quantum.dimensions == (1, 1, 1)
    np.testing.assert_array_equal(quantum.probabilities(task.game.scenario),
                                  np.array(classical.probabilities(), dtype=float))
    checked = evaluate(task, quantum)
    assert checked.value == float(classical.score(task.game))
    assert checked.guarantee == 'numerical_witness'


@pytest.mark.parametrize('tables', [((0,), (0, 0)), ((0, 2), (0, 0)),
                                    ((0, 1.0), (0, 0)), ((False, 1), (0, 0)),
                                    ((0, 1),)])
def test_invalid_deterministic_tables_rejected(tables):
    with pytest.raises(ValueError):
        evaluate(AnalysisTask(chsh(), ResourceSpec('local')), tables)


def test_witness_interface_and_behavior_expansion_guards():
    strategy = DeterministicStrategy(chsh().scenario, ((0, 0), (0, 0)))
    with pytest.raises(ValueError, match='max_entries'):
        strategy.probabilities(max_entries=1)
    with pytest.raises(ValueError, match='differ'):
        strategy.probabilities(ghz().scenario)
    with pytest.raises(ValueError, match='differ'):
        evaluate(AnalysisTask(ghz(), ResourceSpec('local')), strategy)


def test_fixed_state_cannot_be_bypassed_by_exact_response_table_embedding():
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', state_constraint='product',
                        fixed_state=FixedState(np.eye(4)/4, (2, 2))))
    with pytest.raises(NotImplementedError):
        evaluate(task, ((0, 0), (0, 0)))
