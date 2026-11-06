from fractions import Fraction

import pytest

from nonlocalgames import AnalysisTask, Game, Resource, ResourceSpec, Scenario
from nonlocalgames.core.serialization import dump, from_dict, load, to_dict
from nonlocalgames.core.transforms import affine_score, mix, parallel_product, relabel
from nonlocalgames.games import chsh, ghz


def test_exact_chsh_and_schema_roundtrip(tmp_path):
    game = chsh()
    assert game.scenario.entries == 16
    assert sum(game.coefficients) == 2
    path = tmp_path / "game.json"
    dump(game, path)
    assert load(path) == game
    data = to_dict(game)
    data["schema"] = "future"
    with pytest.raises(ValueError):
        from_dict(data)


def test_validation_and_promise():
    for dims in [(), (0,), (-1,), (True,), (1.5,)]:
        with pytest.raises(ValueError):
            Scenario(dims, dims)
    with pytest.raises(ValueError):
        Scenario((2,), (2, 2))
    with pytest.raises(TypeError):
        Game(Scenario((1,), (1,)), (1.0,), (1,))
    with pytest.raises(ValueError):
        Game(Scenario((1,), (1,)), (2,), (1,))
    with pytest.raises(ValueError):
        chsh().reward((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        chsh().reward((0, 0), (0, 2))
    with pytest.raises(ValueError):
        Game.from_rule(Scenario((100, 100), (100, 100)), lambda x, a: 1)
    assert ghz().probability((0, 0, 1)) == 0


def test_no_input_and_signed_score():
    from nonlocalgames.analysis import classical_value
    game = Game.from_rule(Scenario((1, 1), (2, 3)), lambda x, a: a[1] - a[0])
    assert not game.is_acceptance_game
    assert classical_value(game).value == 2


def test_mixture_preserves_score_functional_with_different_sampling():
    left = chsh()
    right = Game.from_rule(left.scenario, lambda x, a: a == (0, 0),
                           distribution={(1, 1): 1})
    mixed = mix([left, right], [Fraction(1, 3), Fraction(2, 3)])
    assert mixed.coefficients == tuple((a + 2*b)/3 for a, b in zip(
        left.coefficients, right.coefficients))
    with pytest.raises(ValueError):
        mix([left], [-1])


def test_relabel_and_affine_invariance():
    from nonlocalgames.analysis import classical_value
    game = chsh()
    changed = relabel(game, ((1, 0), (0, 1)),
                      (((1, 0), (0, 1)), ((1, 0), (1, 0))))
    assert classical_value(changed).value == Fraction(3, 4)
    assert classical_value(affine_score(game, 3, -2)).value == Fraction(1, 4)


def test_parallel_product_does_not_assume_multiplicativity():
    from nonlocalgames.analysis import classical_value
    twice = parallel_product(chsh(), chsh())
    assert twice.scenario == Scenario((4, 4), (4, 4))
    assert sum(twice.distribution) == 1
    assert twice.reward((3, 3), (0, 3)) == 1
    assert classical_value(twice).value == Fraction(10, 16)
    assert classical_value(twice).value > Fraction(3, 4)**2
    with pytest.raises(ValueError):
        parallel_product(chsh(), chsh(), max_entries=10)


def test_resource_dispatch_is_explicit():
    task = AnalysisTask(chsh(), ResourceSpec(Resource.LOCAL))
    assert task.solve().value == Fraction(3, 4)
    with pytest.raises(NotImplementedError):
        AnalysisTask(chsh(), ResourceSpec(Resource.QUANTUM_FINITE)).solve()
    with pytest.raises(ValueError):
        ResourceSpec(Resource.QUANTUM_COMMUTING, (2, 2))
    with pytest.raises(ValueError):
        AnalysisTask(chsh(), ResourceSpec(Resource.QUANTUM_FINITE, (2,)))
