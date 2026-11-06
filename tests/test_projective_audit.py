"""Independent complex contractions and projective-update optimality checks."""
from math import prod

import numpy as np
import pytest

from nonlocalgames import AnalysisTask, Game, ResourceSpec, Scenario
from nonlocalgames.quantum import QuantumStrategy, seesaw
from nonlocalgames.quantum._blocks import effective_operators
from nonlocalgames.quantum._measurements import projective_update, random_projective


def _povms(rng, dimension, inputs, outputs):
    result = []
    for _ in range(inputs):
        raw = rng.normal(size=(dimension, dimension)) + 1j*rng.normal(size=(dimension, dimension))
        unitary, _ = np.linalg.qr(raw)
        weights = rng.uniform(.1, 1, size=(outputs, dimension))
        weights /= weights.sum(axis=0)
        result.append(tuple((unitary * row) @ unitary.conj().T for row in weights))
    return tuple(result)


@pytest.mark.parametrize('dimensions', [(2, 3, 2), (1, 2, 1, 2), (3,)])
def test_effective_operators_predict_all_candidate_local_scores(dimensions):
    rng = np.random.default_rng(105)
    scenario = Scenario((2,)*len(dimensions), tuple(3 if d == 3 else 2 for d in dimensions))
    game = Game.from_rule(
        scenario, lambda x, a: sum((i+1)*(2*a[i]-x[i]) for i in range(len(dimensions)))
        + (1 + sum(x) % 2)*prod(v+1 for v in a))
    size = prod(dimensions)
    raw = rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))
    state = raw @ raw.conj().T
    state /= np.trace(state)
    measurements = tuple(_povms(rng, d, x, a) for d, x, a in
                         zip(dimensions, scenario.input_sizes, scenario.output_sizes))
    coefficients = np.array(game.coefficients, dtype=float).reshape(
        scenario.input_sizes + scenario.output_sizes)
    for party, dimension in enumerate(dimensions):
        effective = effective_operators(coefficients, state, measurements, dimensions, party)
        replacement = _povms(rng, dimension, scenario.input_sizes[party], scenario.output_sizes[party])
        changed = measurements[:party] + (replacement,) + measurements[party+1:]
        direct = QuantumStrategy(state, changed).score(game)
        local = sum(np.trace(e @ h).real for povm, objective in zip(replacement, effective)
                    for e, h in zip(povm, objective))
        assert local == pytest.approx(direct, abs=2e-12)


def test_binary_spectral_update_matches_independent_eigenvalue_formula():
    rng = np.random.default_rng(203)
    for dimension in (1, 2, 5):
        matrices = []
        for _ in range(2):
            raw = rng.normal(size=(dimension, dimension)) + 1j*rng.normal(size=(dimension, dimension))
            matrices.append((raw + raw.conj().T)/2)
        first, second = projective_update(matrices)
        optimum = np.trace(matrices[1]).real + np.maximum(
            np.linalg.eigvalsh(matrices[0] - matrices[1]), 0).sum()
        actual = np.trace(first @ matrices[0] + second @ matrices[1]).real
        assert actual == pytest.approx(optimum, abs=2e-12)
        np.testing.assert_allclose(first @ first, first, atol=1e-12)
        np.testing.assert_allclose(first @ second, 0, atol=1e-12)
        np.testing.assert_allclose(first + second, np.eye(dimension), atol=1e-12)
    zero, full = projective_update((-np.eye(3), np.eye(3)))
    np.testing.assert_array_equal(zero, np.zeros((3, 3)))
    np.testing.assert_array_equal(full, np.eye(3))
    np.testing.assert_array_equal(projective_update((np.eye(3),))[0], np.eye(3))


def test_projective_initializer_preserves_dimension_and_singleton_outcomes():
    for dimension in (1, 2, 4):
        for outputs in (1, 2):
            povm = random_projective(np.random.default_rng(17), dimension, outputs)
            assert len(povm) == outputs
            np.testing.assert_allclose(sum(povm), np.eye(dimension), atol=1e-12)
            for effect in povm:
                assert effect.shape == (dimension, dimension)
                np.testing.assert_allclose(effect @ effect, effect, atol=1e-12)
    with pytest.raises(NotImplementedError):
        random_projective(np.random.default_rng(17), 3, 3)


def test_single_party_search_and_nonbinary_projective_rejection():
    game = Game.from_rule(Scenario((2,), (2,)), lambda x, a: int(a == x))
    task = AnalysisTask(game, ResourceSpec('quantum_finite', (1,), measurement_class='projective'))
    assert 'seesaw' in task.capabilities()
    result = task.solve(method='seesaw', restarts=1, max_iterations=2, seed=2)
    assert result.value == 1
    assert result.witness.dimensions == (1,)
    assert set(result.diagnostics['solver_statuses']) == {'analytic_projective'}
    unsupported = Game.from_rule(Scenario((1,), (3,)), lambda x, a: a[0])
    task = AnalysisTask(unsupported, ResourceSpec('quantum_finite', (3,), measurement_class='projective'))
    assert 'seesaw' not in task.capabilities()
    with pytest.raises(NotImplementedError):
        seesaw(unsupported, (3,), measurement_class='projective')
