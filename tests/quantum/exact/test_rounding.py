"""Numerical reconstruction produces a new exactly feasible strategy, never a proximity proof."""
from fractions import Fraction
from math import prod

import numpy as np
import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import bilocal_entanglement_swapping
from nonlocalgames.quantum import QuantumStrategy, chsh_strategy
from nonlocalgames.quantum.exact.algebra import add, identity, scale
from nonlocalgames.quantum.exact.budgets import ExactBudget, ExactBudgetExceeded
from nonlocalgames.quantum.exact.matrix import verify_psd_complex
from nonlocalgames.quantum.exact.rounding import rationalize_strategy


def _check_exact_realization(strategy, budget=None):
    assert strategy.state.trace() == (Fraction(1), Fraction(0))
    verify_psd_complex(strategy.state, budget=budget)
    for dimension, party in zip(strategy.dimensions, strategy.measurements):
        unit = identity(dimension, budget=budget)
        for povm in party:
            total = scale(unit, 0, budget=budget)
            for effect in povm:
                verify_psd_complex(effect, budget=budget)
                total = add(total, effect, budget=budget)
            assert total.real == unit.real and total.imag == unit.imag


def test_rounding_keeps_source_immutable_and_checks_exact_feasibility():
    source = chsh_strategy()
    arrays = (source.state,) + tuple(e for p in source.measurements for povm in p for e in povm)
    before = tuple(array.tobytes() for array in arrays)
    result = rationalize_strategy(source)
    _check_exact_realization(result.strategy)
    from nonlocalgames.games import chsh
    assert result.strategy.score(chsh()) > Fraction(3, 4)
    assert result.strategy.dimensions == source.dimensions
    assert tuple(array.tobytes() for array in arrays) == before
    assert not result.diagnostics['closeness_or_optimality_claimed']
    assert set(result.diagnostics['restrictions_not_preserved']) == {
        'fixed_state', 'ppt', 'projective', 'product', 'network'}


def test_zero_bisection_steps_retains_the_feasible_identity_endpoint():
    vector = np.array([3., 1.])/np.sqrt(10)
    state = np.outer(vector, vector)
    source = QuantumStrategy(state, (((state, np.eye(2)-state),),))
    result = rationalize_strategy(source, max_denominator=2, repair_steps=0)
    _check_exact_realization(result.strategy)
    assert result.diagnostics['state']['mixing_weight'] == 1
    assert result.diagnostics['measurements'][0][0]['mixing_weight'] == 1
    assert result.diagnostics['state']['bisection_steps'] == 0
    half = ((Fraction(1, 2), Fraction(0)), (Fraction(0), Fraction(1, 2)))
    assert result.strategy.state.real == half
    assert all(effect.real == half for effect in result.strategy.measurements[0][0])


def test_complex_mixed_nonbinary_strategy_reconstructs_exactly_without_floating_eigenchecks(monkeypatch):
    rng = np.random.default_rng(18)
    dimensions, inputs, outputs = (2, 2, 1), (2, 1, 1), (3, 2, 2)
    size = prod(dimensions)
    raw = rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))
    state = raw @ raw.conj().T
    state /= np.trace(state)
    parties = []
    for dimension, questions, answers in zip(dimensions, inputs, outputs):
        party = []
        for _ in range(questions):
            raw = rng.normal(size=(dimension, dimension)) + 1j*rng.normal(size=(dimension, dimension))
            unitary, _ = np.linalg.qr(raw)
            diagonal = rng.uniform(.1, 1, size=(answers, dimension))
            diagonal /= diagonal.sum(axis=0)
            party.append(tuple((unitary*row) @ unitary.conj().T for row in diagonal))
        parties.append(tuple(party))
    source = QuantumStrategy(state, tuple(parties))

    def forbidden(*args, **kwargs):
        pytest.fail('Floating eigenvalues must not establish exact feasibility')

    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    monkeypatch.setattr(np.linalg, 'eigvalsh', forbidden)
    result = rationalize_strategy(source, max_denominator=100, repair_steps=12)
    _check_exact_realization(result.strategy)
    scenario = Scenario(inputs, outputs)
    probabilities = result.strategy.probabilities(scenario)
    width = prod(outputs)
    assert all(isinstance(value, Fraction) and value >= 0 for value in probabilities)
    assert all(sum(probabilities[start:start+width]) == 1 for start in range(0, len(probabilities), width))


def test_trace_and_completeness_defects_are_corrected_before_psd_repair():
    source = QuantumStrategy(np.array([[1.01]]), (((np.array([[.2]]), np.array([[.7]])),),), tolerance=.2)
    result = rationalize_strategy(source, max_denominator=100, repair_steps=3)
    _check_exact_realization(result.strategy)
    assert result.diagnostics['state']['trace_identity_shift'] == -Fraction(1, 100)
    assert result.diagnostics['measurements'][0][0]['completeness_correction_max_component'] == Fraction(1, 20)
    assert result.strategy.state.real == ((Fraction(1),),)
    assert tuple(effect.real[0][0] for effect in result.strategy.measurements[0][0]) == (Fraction(1, 4), Fraction(3, 4))


def test_repair_budgets_fail_before_rational_conversion(monkeypatch):
    import nonlocalgames.quantum.exact.rounding as rounding
    monkeypatch.setattr(rounding, '_round_hermitian', lambda *a, **k: pytest.fail('budget must fail first'))
    source = chsh_strategy()
    for budget in (ExactBudget(max_joint_dimension=3), ExactBudget(max_psd_size=7),
                   ExactBudget(max_matrix_entries=1), ExactBudget(max_parties=1)):
        with pytest.raises(ExactBudgetExceeded):
            rationalize_strategy(source, budget=budget)


def test_bisection_does_not_swallow_exact_budget_failure(monkeypatch):
    import nonlocalgames.quantum.exact.rounding as rounding

    def exhausted(*args, **kwargs):
        raise ExactBudgetExceeded('synthetic exact arithmetic budget')

    monkeypatch.setattr(rounding, 'verify_psd_complex', exhausted)
    with pytest.raises(ExactBudgetExceeded, match='synthetic'):
        rationalize_strategy(chsh_strategy())


def test_network_witness_and_implicit_descriptors_are_rejected():
    network = bilocal_entanglement_swapping()
    for source in (network, {'state': chsh_strategy().state}, None):
        with pytest.raises(TypeError, match='plain numerical QuantumStrategy'):
            rationalize_strategy(source)


@pytest.mark.parametrize('options', [
    {'max_denominator': 0}, {'max_denominator': True}, {'max_denominator': 1.5},
    {'repair_steps': -1}, {'repair_steps': True}, {'repair_steps': 1.5},
])
def test_reconstruction_parameters_are_explicit_positive_or_nonnegative_integers(options):
    with pytest.raises(ValueError):
        rationalize_strategy(chsh_strategy(), **options)


def test_bisection_keeps_the_exact_feasible_side_of_a_known_qubit_threshold():
    vector = np.array([3., 1.])/np.sqrt(10)
    state = np.outer(vector, vector)
    source = QuantumStrategy(state, (((state, np.eye(2)-state),),))
    steps = 12
    result = rationalize_strategy(source, max_denominator=2, repair_steps=steps)
    weight = result.diagnostics['state']['mixing_weight']
    lower = weight-Fraction(1, 2**steps)
    assert 0 < weight < 1
    assert 2*weight*weight-4*weight+1 <= 0
    assert 2*lower*lower-4*lower+1 >= 0
    assert result.diagnostics['state']['bisection_steps'] == steps
    assert result.diagnostics['measurements'][0][0]['mixing_weight'] == weight
    _check_exact_realization(result.strategy)


def test_question_dependent_output_counts_are_not_silently_padded(monkeypatch):
    import nonlocalgames.quantum.exact.rounding as rounding
    source = QuantumStrategy(np.ones((1, 1)), (((np.ones((1, 1)),),
                              (np.ones((1, 1))/2, np.ones((1, 1))/2)),))
    monkeypatch.setattr(rounding, '_round_hermitian', lambda *a, **k: pytest.fail('interface must fail first'))
    with pytest.raises(ValueError, match='fixed output alphabet'):
        rationalize_strategy(source)


def test_rounded_exact_strategy_survives_strict_codec_roundtrip():
    import json
    from nonlocalgames.games import chsh
    from nonlocalgames.quantum.exact.serialization import strategy_from_dict, strategy_to_dict
    result = rationalize_strategy(chsh_strategy(), max_denominator=1000, repair_steps=16)
    payload = json.loads(json.dumps(strategy_to_dict(result.strategy)))
    decoded = strategy_from_dict(payload)
    assert decoded.dimensions == result.strategy.dimensions
    assert decoded.state == result.strategy.state
    assert decoded.measurements == result.strategy.measurements
    assert decoded.score(chsh()) == result.strategy.score(chsh())
    assert decoded.validate()['certified']
