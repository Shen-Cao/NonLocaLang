"""Independent full-tensor checks for complex quantum block optimization."""
from fractions import Fraction

import cvxpy as cp
import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.quantum import QuantumStrategy
from nonlocalgames.quantum._blocks import normalize_povm, optimize_party


def _fixture():
    rng = np.random.default_rng(2026)
    dimensions = (2, 3)
    scenario = Scenario((2, 3), (3, 2))
    payoff = tuple(Fraction(int(v), 7) for v in rng.integers(-5, 8, scenario.entries))
    game = Game(scenario, (Fraction(1, 6),)*6, payoff)
    factor = rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6))
    state = factor @ factor.conj().T
    state /= np.trace(state)
    measurements = []
    for d, nx, na in zip(dimensions, scenario.input_sizes, scenario.output_sizes):
        party = []
        for _ in range(nx):
            gs = rng.normal(size=(na, d, d)) + 1j*rng.normal(size=(na, d, d))
            party.append(normalize_povm([g @ g.conj().T for g in gs]))
        measurements.append(tuple(party))
    return game, QuantumStrategy(state, tuple(measurements))


@pytest.mark.parametrize('party', [0, 1])
def test_complex_unequal_dimension_update_matches_full_tensor_sdp(party):
    game, original = _fixture()
    coefficients = np.asarray(tuple(map(float, game.coefficients))).reshape(2, 3, 3, 2)
    updated, statuses = optimize_party(coefficients, original.state, original.measurements,
                                       original.dimensions, party, 'SCS', {'eps': 1e-9})
    actual = list(original.measurements)
    actual[party] = updated
    candidate = QuantumStrategy(original.state, tuple(actual), tolerance=1e-7)
    d = original.dimensions[party]
    effects = [[cp.Variable((d, d), hermitian=True) for _ in range(game.scenario.output_sizes[party])]
               for _ in range(game.scenario.input_sizes[party])]
    constraints = [effect >> 0 for povm in effects for effect in povm]
    constraints += [sum(povm) == np.eye(d) for povm in effects]
    score = 0
    for x in game.scenario.inputs:
        for a in game.scenario.outputs:
            alice = effects[x[0]][a[0]] if party == 0 else original.measurements[0][x[0]][a[0]]
            bob = effects[x[1]][a[1]] if party == 1 else original.measurements[1][x[1]][a[1]]
            if party == 0:
                joint = cp.bmat([[alice[i, j] * bob for j in range(d)] for i in range(d)])
            else:
                joint = cp.kron(alice, bob)
            score += float(game.coefficient(x, a))*cp.real(cp.trace(original.state @ joint))
    problem = cp.Problem(cp.Maximize(score), constraints)
    problem.solve(solver='SCS', eps=1e-9)
    assert all(status == 'optimal' for status in statuses)
    assert problem.status == 'optimal'
    assert candidate.score(game) == pytest.approx(problem.value, abs=2e-7)
    assert candidate.score(game) >= original.score(game) - 1e-8
