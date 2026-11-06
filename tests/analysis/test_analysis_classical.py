from fractions import Fraction
from itertools import product
from random import Random

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import classical_value
from nonlocalgames.games import chsh, ghz, gyni, magic_square


@pytest.mark.parametrize("constructor, expected", [
    (chsh, Fraction(3, 4)), (ghz, Fraction(3, 4)),
    (gyni, Fraction(1, 4)), (magic_square, Fraction(8, 9)),
])
def test_exact_benchmarks_and_witness(constructor, expected):
    game = constructor()
    result = classical_value(game)
    assert result.value == expected
    assert result.guarantee == "exact"
    witness_value = sum(game.coefficient(x, tuple(t[x[i]] for i, t in enumerate(result.witness)))
                        for x in game.inputs)
    assert witness_value == expected


def test_best_response_matches_independent_exhaustive_search():
    s = Scenario((2, 1, 2), (2, 3, 2))
    game = Game.from_rule(s, lambda x, a: Fraction((sum(a) + 2*x[0] - a[2]*x[2]) % 5 - 2, 3))
    strategies = [tuple(product(range(na), repeat=nx)) for nx, na in zip(s.input_sizes, s.output_sizes)]
    brute = max(sum(game.coefficient(x, tuple(t[x[i]] for i, t in enumerate(strategy)))
                    for x in game.inputs) for strategy in product(*strategies))
    assert classical_value(game).value == brute


def test_seeded_games_match_exhaustive_search():
    rng = Random(937)
    for _ in range(12):
        parties = rng.choice((1, 2, 3))
        s = Scenario(tuple(rng.choice((1, 2)) for _ in range(parties)),
                     tuple(rng.choice((1, 2, 3)) for _ in range(parties)))
        raw_weights = [rng.randrange(1, 5) for _ in s.inputs]
        game = Game(s, tuple(Fraction(w, sum(raw_weights)) for w in raw_weights),
                    tuple(Fraction(rng.randrange(-4, 5), 3) for _ in range(s.entries)))
        strategies = [tuple(product(range(na), repeat=nx))
                      for nx, na in zip(s.input_sizes, s.output_sizes)]
        brute = max(sum(game.coefficient(x, tuple(t[x[i]] for i, t in enumerate(strategy)))
                        for x in game.inputs) for strategy in product(*strategies))
        assert classical_value(game).value == brute


def test_single_party_and_guard():
    game = Game.from_rule(Scenario((3,), (2,)), lambda x, a: int(a[0] == x[0] % 2))
    assert classical_value(game, max_strategies=1).value == 1
    with pytest.raises(ValueError, match="limit"):
        classical_value(chsh(), max_strategies=3)
    with pytest.raises(ValueError):
        classical_value(chsh(), max_strategies=True)
