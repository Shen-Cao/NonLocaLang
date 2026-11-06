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


def relabel(game, input_permutations, output_permutations):
    """Maps new labels to old labels locally; output maps may depend on new input.

    input_permutations[i][x_new] = x_old;
    output_permutations[i][x_new][a_new] = a_old.
    """
    s = game.scenario
    ip = tuple(tuple(p) for p in input_permutations)
    op = tuple(tuple(tuple(p) for p in party) for party in output_permutations)
    if len(ip) != s.parties or len(op) != s.parties:
        raise ValueError("Provide permutations for each party")
    for i, (nx, na) in enumerate(zip(s.input_sizes, s.output_sizes)):
        if sorted(ip[i]) != list(range(nx)) or len(op[i]) != nx:
            raise ValueError("Invalid input permutation or output map shape")
        if any(sorted(p) != list(range(na)) for p in op[i]):
            raise ValueError("Invalid output permutation")

    def old_x(x):
        return tuple(ip[i][x[i]] for i in range(s.parties))

    def old_a(x, a):
        return tuple(op[i][x[i]][a[i]] for i in range(s.parties))
    return Game.from_rule(s, lambda x, a: game.reward(old_x(x), old_a(x, a)),
                          distribution={x: game.probability(old_x(x)) for x in s.inputs},
                          name=game.name + " (relabelled)")


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
