"""Search witnesses against known optima and validate physical outputs."""

from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.quantum import seesaw


def _game(scenario, payoff):
    return Game(
        scenario,
        (Fraction(1, len(scenario.inputs)),) * len(scenario.inputs),
        tuple(Fraction(payoff(x, a)) for x in scenario.inputs for a in scenario.outputs),
        "test game",
    )


def test_seesaw_finds_chsh_advantage():
    game = _game(Scenario((2, 2), (2, 2)), lambda x, a: (a[0] ^ a[1]) == x[0] * x[1])
    result = seesaw(game, restarts=2, max_iterations=12, seed=7, solver_options={"eps": 1e-7})
    assert result.value == pytest.approx(0.5 + np.sqrt(2) / 4, abs=1e-5)
    assert result.value == result.strategy.score(game)
    assert result.history[result.iterations] == result.value
    assert result.value == max(result.restart_scores)
    assert result.strategy.validate()["normalization_residual"] < 1e-7
    assert set(result.solver_statuses) <= {"optimal", "optimal_inaccurate"}


def test_three_outcome_product_game_and_seed_reproducibility():
    game = _game(Scenario((1, 1), (3, 3)), lambda x, a: a == (1, 2))
    kwargs = {"dimensions": (1, 1), "restarts": 1, "max_iterations": 4, "seed": 8, "solver_options": {"eps": 1e-8}}
    first, second = seesaw(game, **kwargs), seesaw(game, **kwargs)
    assert first.value == pytest.approx(1, abs=1e-6)
    np.testing.assert_allclose(first.history, second.history, atol=1e-12)
    assert first.strategy.dimensions == (1, 1)


@pytest.mark.parametrize("kwargs", [
    {"restarts": 0}, {"max_iterations": 0}, {"max_iterations": 1.5},
    {"dimensions": (0, 2)}, {"dimensions": (2,)}, {"dimensions": (2.0, 2)},
    {"seed": -1}, {"convergence_tolerance": 0}, {"validation_tolerance": np.nan},
    {"solver_options": {"solver": "SCS"}},
])
def test_invalid_search_configuration(kwargs):
    game = _game(Scenario((1, 1), (2, 2)), lambda x, a: True)
    with pytest.raises(ValueError):
        seesaw(game, **kwargs)


def test_reject_mismatched_multipartite_dimensions():
    game = _game(Scenario((1, 1, 1), (2, 2, 2)), lambda x, a: True)
    with pytest.raises(ValueError, match="per party"):
        seesaw(game, dimensions=(2, 2))
