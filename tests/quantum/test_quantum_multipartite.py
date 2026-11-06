"""Multipartite and projective searches checked against independent objectives."""
from fractions import Fraction
from functools import reduce

import cvxpy as cp
import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.games import ghz
from nonlocalgames.quantum import FixedState, QuantumStrategy, ghz_strategy, seesaw, state_diagnostics
from nonlocalgames.quantum._blocks import effective_operators, normalize_povm, optimize_party
from nonlocalgames.quantum._measurements import projective_update


def _assert_projective(strategy):
    for party in strategy.measurements:
        for povm in party:
            for i, effect in enumerate(povm):
                for j, other in enumerate(povm):
                    np.testing.assert_allclose(effect @ other, effect if i == j else 0, atol=1e-10)


@pytest.mark.parametrize('measurement_class', ['povm', 'projective'])
def test_three_party_ghz_search_reaches_perfect_score(measurement_class):
    result = seesaw(ghz(), measurement_class=measurement_class, seed=7, restarts=2,
                     max_iterations=12, solver_options={'eps': 1e-8})
    assert result.value == pytest.approx(1, abs=2e-6)
    assert result.strategy.dimensions == (2, 2, 2)
    assert result.value == result.strategy.score(ghz())
    assert result.measurement_class == measurement_class
    if measurement_class == 'projective':
        _assert_projective(result.strategy)
        assert set(result.solver_statuses) == {'analytic_projective'}


def test_fixed_ghz_state_and_projective_initialization_are_reproducible():
    fixed = FixedState(ghz_strategy().state, (2, 2, 2))
    options = dict(fixed_state=fixed, measurement_class='projective', seed=7,
                   restarts=2, max_iterations=15)
    first, second = seesaw(ghz(), **options), seesaw(ghz(), **options)
    assert first.value == pytest.approx(1, abs=1e-8)
    np.testing.assert_array_equal(first.strategy.state, fixed.state)
    np.testing.assert_allclose(first.history, second.history, atol=1e-12)
    assert not first.state_updates
    _assert_projective(first.strategy)


def test_multipartite_ppt_cuts_survive_state_and_measurement_updates():
    cuts = ((0,), (1,), (2,))
    result = seesaw(ghz(), ppt_cuts=cuts, seed=7, restarts=1, max_iterations=5,
                     solver_options={'eps': 1e-8})
    assert result.value == pytest.approx(.75, abs=1e-6)
    state_diagnostics(result.strategy.state, (2, 2, 2), ppt_cuts=cuts)
    assert result.ppt_cuts == ((0,), (0, 1), (0, 2))


def test_single_party_and_degenerate_projectors():
    binary = Game.from_rule(Scenario((2,), (2,)), lambda x, a: a[0] == x[0])
    result = seesaw(binary, dimensions=(1,), measurement_class='projective', seed=3,
                     restarts=1, max_iterations=3)
    assert result.value == 1
    _assert_projective(result.strategy)
    for difference, expected in [(np.eye(3), np.eye(3)), (-np.eye(3), np.zeros((3, 3)))]:
        first, second = projective_update((difference, np.zeros((3, 3))))
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(first+second, np.eye(3))
    unary = Game.from_rule(Scenario((2, 1, 1), (1, 2, 1)), lambda x, a: a[1] == 1)
    result = seesaw(unary, measurement_class='projective', seed=5, restarts=1, max_iterations=3)
    assert result.value == pytest.approx(1, abs=1e-12)
    _assert_projective(result.strategy)


def test_projective_search_rejects_unsupported_outputs_and_costs():
    game = Game.from_rule(Scenario((1, 1, 1), (2, 3, 2)), lambda x, a: True)
    with pytest.raises(NotImplementedError, match='one or two outputs'):
        seesaw(game, measurement_class='projective')
    with pytest.raises(ValueError, match='measurement_class'):
        seesaw(game, measurement_class='unknown')
    with pytest.raises(ValueError, match='max_update_terms'):
        seesaw(ghz(), max_update_terms=100)


@pytest.mark.parametrize('party', [0, 1, 2])
def test_unequal_complex_three_party_update_matches_full_tensor_sdp(party):
    rng = np.random.default_rng(91)
    dimensions, scenario = (2, 3, 2), Scenario((1, 2, 1), (2, 3, 2))
    payoff = tuple(Fraction(int(v), 5) for v in rng.integers(-3, 7, scenario.entries))
    game = Game(scenario, (Fraction(1, 2),)*2, payoff)
    gaussian = rng.normal(size=(12, 12)) + 1j*rng.normal(size=(12, 12))
    rho = gaussian @ gaussian.conj().T
    rho /= np.trace(rho)
    measurements = []
    for dimension, nx, na in zip(dimensions, scenario.input_sizes, scenario.output_sizes):
        local = []
        for _ in range(nx):
            gs = rng.normal(size=(na, dimension, dimension)) + 1j*rng.normal(size=(na, dimension, dimension))
            local.append(normalize_povm([g @ g.conj().T for g in gs]))
        measurements.append(tuple(local))
    original = QuantumStrategy(rho, tuple(measurements))
    coefficients = np.asarray(tuple(map(float, game.coefficients))).reshape(1, 2, 1, 2, 3, 2)
    effective = effective_operators(coefficients, rho, original.measurements, dimensions, party)
    local_score = sum(np.trace(e @ h).real for povm, hs in zip(original.measurements[party], effective)
                      for e, h in zip(povm, hs))
    assert local_score == pytest.approx(original.score(game), abs=1e-12)
    updated, _ = optimize_party(coefficients, rho, original.measurements, dimensions, party,
                                'SCS', {'eps': 1e-9})
    candidate_measurements = list(original.measurements)
    candidate_measurements[party] = updated
    candidate = QuantumStrategy(rho, tuple(candidate_measurements), tolerance=1e-7)
    d = dimensions[party]
    effects = [[cp.Variable((d, d), hermitian=True) for _ in range(scenario.output_sizes[party])]
               for _ in range(scenario.input_sizes[party])]
    constraints = [e >> 0 for povm in effects for e in povm]
    constraints += [sum(povm) == np.eye(d) for povm in effects]
    reference_coefficients = [[np.zeros((d, d), dtype=complex) for _ in range(scenario.output_sizes[party])]
                              for _ in range(scenario.input_sizes[party])]
    for x in scenario.inputs:
        for a in scenario.outputs:
            left = reduce(np.kron, (original.measurements[i][x[i]][a[i]] for i in range(party)), np.ones((1, 1)))
            right = reduce(np.kron, (original.measurements[i][x[i]][a[i]] for i in range(party+1, 3)), np.ones((1, 1)))
            for i in range(d):
                for j in range(d):
                    unit = np.zeros((d, d))
                    unit[i, j] = 1
                    joint = np.kron(np.kron(left, unit), right)
                    reference_coefficients[x[party]][a[party]][i, j] += float(game.coefficient(x, a))*np.trace(rho @ joint)
    objective = sum(cp.real(cp.sum(cp.multiply(effect, coefficient)))
                    for povm, cs in zip(effects, reference_coefficients) for effect, coefficient in zip(povm, cs))
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver='SCS', eps=1e-9)
    assert problem.status == 'optimal'
    assert candidate.score(game) == pytest.approx(problem.value, abs=3e-7)


def test_measurement_memory_guard_precedes_random_initialization(monkeypatch):
    game = Game.from_rule(Scenario((17,), (1,)), lambda x, a: True)
    monkeypatch.setattr(np.random, "default_rng", lambda *args: pytest.fail("must reject before initialization"))
    with pytest.raises(ValueError, match="max_measurement_entries"):
        seesaw(game, dimensions=(256,))


@pytest.mark.parametrize("options", [
    {"max_measurement_entries": 0}, {"max_measurement_entries": True},
    {"validation_tolerance": np.bool_(True)}, {"convergence_tolerance": np.bool_(True)},
])
def test_new_search_guards_reject_invalid_configuration(options):
    with pytest.raises(ValueError):
        seesaw(ghz(), **options)
