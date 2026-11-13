"""Verifier transformations with explicit operational semantics."""
from .game import Game, Scenario, rational


def mix(games, weights, *, name="Hidden mixture"):
    """Choose a verifier privately; expose only each player's ordinary input.

    The returned payoff is the posterior average given the joint input. No
    component label is sent to players; this is not a public direct sum.
    """
    games, weights = tuple(games), tuple(map(rational, weights))
    if not games or len(games) != len(weights):
        raise ValueError("Provide one weight per game")
    if any(w < 0 for w in weights) or sum(weights) != 1:
        raise ValueError("Mixture weights must be nonnegative and sum to one")
    s = games[0].scenario
    if any(g.scenario != s for g in games):
        raise ValueError("Hidden mixture needs a common interface")
    mu = {x: sum(w*g.probability(x) for w, g in zip(weights, games)) for x in s.inputs}

    def score(x, a):
        return (sum(w*g.coefficient(x, a) for w, g in zip(weights, games)) / mu[x]
                if mu[x] else 0)
    return Game.from_rule(s, score, distribution=mu, name=name)


def relabel(game, input_permutations, output_permutations, *, max_entries=1_000_000,
            max_map_entries=1_000_000):
    """Relabel local interfaces through a checked new-to-old bijection descriptor.

    This convenience wrapper appends " (relabelled)" to the display name.
    Use equivalence.Relabelling and relabel_game for party permutations,
    inverse/composition, resource transport, or an unchanged display name.
    """
    from ..equivalence.maps import Relabelling
    from ..equivalence.finite import relabel_game
    mapping = Relabelling(game.scenario, input_maps=input_permutations,
                           output_maps=output_permutations, max_map_entries=max_map_entries)
    return relabel_game(game, mapping, max_entries=max_entries, name=game.name + " (relabelled)")


def parallel_product(left, right, *, max_entries=1_000_000):
    """Same parties receive both inputs and return both outputs; multiply rewards.

    Joint strategies across repetitions remain allowed. This construction does
    not imply multiplicativity of any optimized value.
    """
    left_scenario, right_scenario = left.scenario, right.scenario
    if left_scenario.parties != right_scenario.parties:
        raise ValueError("Parallel product requires the same parties")
    s = Scenario(tuple(a*b for a, b in zip(left_scenario.input_sizes, right_scenario.input_sizes)),
                 tuple(a*b for a, b in zip(left_scenario.output_sizes, right_scenario.output_sizes)))
    if s.entries > max_entries:
        raise ValueError(f"Product needs {s.entries} dense entries; limit {max_entries}")

    def split(profile, sizes):
        return tuple(v // n for v, n in zip(profile, sizes)), tuple(
            v % n for v, n in zip(profile, sizes))

    def score(x, a):
        xl, xr = split(x, right_scenario.input_sizes)
        al, ar = split(a, right_scenario.output_sizes)
        return left.reward(xl, al) * right.reward(xr, ar)

    mu = {}
    for x in s.inputs:
        xl, xr = split(x, right_scenario.input_sizes)
        mu[x] = left.probability(xl) * right.probability(xr)
    return Game.from_rule(s, score, distribution=mu, name=left.name + " x " + right.name,
                          max_entries=max_entries)


def affine_score(game, scale, offset):
    """Return scale*score+offset. Positive scale preserves score maximizers."""
    scale, offset = rational(scale), rational(offset)
    return Game(game.scenario, game.distribution,
                tuple(scale*v + offset for v in game.payoff), game.name + " (affine)")
