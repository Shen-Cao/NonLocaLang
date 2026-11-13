"""Finite relabelling preserves full verifier data and exact transported scores."""
from fractions import Fraction

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import classical_value
from nonlocalgames.classical import DeterministicStrategy
from nonlocalgames.equivalence.finite import (relabel_behavior, relabel_deterministic_strategy,
                                             relabel_game)
from nonlocalgames.equivalence.maps import Relabelling, RelabellingBudgetExceeded
from nonlocalgames.games import chsh
from nonlocalgames.networks.behavior import RationalBehavior


def _example():
    scenario = Scenario((2, 3, 1), (3, 2, 4))
    weights = (1, 0, 2, 3, 0, 4)
    distribution = {x: Fraction(w, 10) for x, w in zip(scenario.inputs, weights)}
    game = Game.from_rule(scenario,
        lambda x, a: 10*x[0]-3*x[1] + sum((i+1)*v for i, v in enumerate(a))
        + Fraction(a[0]*a[2]-a[1], 7) + (1000 if distribution[x] == 0 else 0),
        distribution=distribution, name='signed promised verifier')
    mapping = Relabelling(scenario, (1, 2, 0), ((2, 0, 1), (0,), (1, 0)),
        (((1, 0), (0, 1), (1, 0)), ((2, 0, 3, 1),), ((1, 2, 0), (2, 0, 1))))
    first = DeterministicStrategy(scenario, ((2, 1), (1, 0, 1), (3,)))
    second = DeterministicStrategy(scenario, ((0, 2), (0, 1, 0), (1,)))
    return game, mapping, first, second


def test_game_transport_preserves_sampling_rewards_and_unused_rows_separately():
    game, mapping, _, _ = _example()
    transformed = relabel_game(game, mapping)
    assert transformed.name == game.name
    for x in transformed.inputs:
        source_x = mapping.source_input(x)
        assert transformed.probability(x) == game.probability(source_x)
        for a in transformed.outputs:
            assert transformed.reward(x, a) == game.reward(source_x, mapping.source_output(x, a))
    assert relabel_game(transformed, mapping.inverse()) == game
    assert relabel_game(game, Relabelling(game.scenario)) == game
    assert relabel_game(game, mapping, name='explicit target name').name == 'explicit target name'
    assert classical_value(transformed).value == classical_value(game).value


def test_behavior_and_local_strategy_scores_transport_exactly():
    game, mapping, first, second = _example()
    behavior = RationalBehavior(game.scenario, tuple((a+2*b)/3 for a, b in
                               zip(first.probabilities(), second.probabilities())))
    transformed_game = relabel_game(game, mapping)
    transformed_behavior = relabel_behavior(behavior, mapping, max_entries=200)
    assert transformed_behavior.max_entries == 200
    original_score = sum(c*p for c, p in zip(game.coefficients, behavior.probabilities))
    target_score = sum(c*p for c, p in zip(transformed_game.coefficients, transformed_behavior.probabilities))
    assert original_score == target_score
    assert relabel_behavior(transformed_behavior, mapping.inverse()) == behavior
    for strategy in (first, second):
        transported = relabel_deterministic_strategy(strategy, mapping)
        assert transported.score(transformed_game) == strategy.score(game)
        point = RationalBehavior(game.scenario, strategy.probabilities())
        assert relabel_behavior(point, mapping).probabilities == transported.probabilities()
        assert relabel_deterministic_strategy(transported, mapping.inverse()) == strategy


def test_sequential_finite_transport_agrees_with_map_composition():
    game, first, strategy, _ = _example()
    second = Relabelling(first.target, (2, 0, 1), ((0, 1), (1, 2, 0), (0,)),
        (((2, 1, 0), (0, 2, 1)), ((0, 1), (1, 0), (0, 1)), ((1, 3, 0, 2),)))
    assert relabel_game(relabel_game(game, first), second) == relabel_game(game, first.then(second))
    assert relabel_deterministic_strategy(relabel_deterministic_strategy(strategy, first), second) == \
        relabel_deterministic_strategy(strategy, first.then(second))


def test_table_budget_precedes_profile_materialization(monkeypatch):
    game = chsh()
    mapping = Relabelling(game.scenario)
    behavior = RationalBehavior(game.scenario, (Fraction(1, 4),)*16)

    def forbidden(*args):
        pytest.fail('table-size guards must precede Cartesian profile expansion')

    monkeypatch.setattr(Scenario, 'inputs', property(forbidden))
    monkeypatch.setattr(Scenario, 'outputs', property(forbidden))
    with pytest.raises(RelabellingBudgetExceeded, match='max_entries'):
        relabel_game(game, mapping, max_entries=15)
    with pytest.raises(RelabellingBudgetExceeded, match='max_entries'):
        relabel_behavior(behavior, mapping, max_entries=15)


def test_transport_rejects_wrong_source_interfaces_and_invalid_budgets():
    game, mapping, strategy, _ = _example()
    wrong = Relabelling(chsh().scenario)
    for transform, value in ((relabel_game, game), (relabel_deterministic_strategy, strategy),
                              (relabel_behavior, RationalBehavior(game.scenario, strategy.probabilities()))):
        with pytest.raises(ValueError, match='source'):
            transform(value, wrong)
    for budget in (0, True, 1.5):
        with pytest.raises(ValueError):
            relabel_game(game, mapping, max_entries=budget)
    with pytest.raises(TypeError):
        relabel_game(game, mapping, name=123)
