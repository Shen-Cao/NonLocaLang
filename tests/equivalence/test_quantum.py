"""Quantum transports preserve tensor order, exact scores, and numerical data bits."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.equivalence.finite import relabel_game
from nonlocalgames.equivalence.maps import Relabelling, RelabellingBudgetExceeded
from nonlocalgames.equivalence.quantum import (NumericalRelabellingError, _permute_numeric_density,
                                              relabel_exact_strategy, relabel_quantum_strategy)
from nonlocalgames.quantum import QuantumStrategy, chsh_strategy
from nonlocalgames.quantum.exact import ExactBudget, ExactBudgetExceeded, ExactQuantumStrategy, RationalMatrix
from nonlocalgames.quantum.exact.algebra import identity, kron, scale, subtract


def _fixture():
    policy = ExactBudget(max_joint_dimension=8, max_psd_size=16, max_matrix_entries=1000)
    plus_i = RationalMatrix(((Fraction(1, 2), 0), (0, Fraction(1, 2))),
                             ((0, Fraction(-1, 2)), (Fraction(1, 2), 0)), budget=policy)
    bob = RationalMatrix(((Fraction(1, 6), 0, 0), (0, Fraction(1, 3), 0),
                           (0, 0, Fraction(1, 2))), budget=policy)
    z0 = RationalMatrix(((1, 0), (0, 0)), budget=policy)
    alice = ((z0, subtract(identity(2, budget=policy), z0, budget=policy)),
             (plus_i, subtract(identity(2, budget=policy), plus_i, budget=policy)))
    bob_povm = tuple(RationalMatrix(tuple(tuple(int(i == j == a) for j in range(3)) for i in range(3)),
                                    budget=policy) for a in range(3))
    charlie = ((identity(1, budget=policy),),)*3
    state = kron(kron(plus_i, bob, budget=policy), identity(1, budget=policy), budget=policy)
    witness = ExactQuantumStrategy(state, (alice, (bob_povm,), charlie), (2, 3, 1), budget=policy)
    mapping = Relabelling(witness.scenario, (1, 2, 0), ((0,), (2, 0, 1), (1, 0)),
                          (((2, 0, 1),), ((0,), (0,), (0,)), ((1, 0), (0, 1))))
    weights = (1, 0, 2, 0, 3, 0)
    game = Game.from_rule(witness.scenario, lambda x, a: Fraction(2*a[0]-a[1]+x[2]-x[0], 5),
                          distribution={x: Fraction(weights[i], 6) for i, x in enumerate(witness.scenario.inputs)})
    return witness, mapping, game


def _numeric(matrix):
    value = np.empty((matrix.dimension, matrix.dimension), dtype=complex)
    value.real = np.array(matrix.real, dtype=float)
    value.imag = np.array(matrix.imag, dtype=float)
    # Numerical sign bits are data and must survive a relabelling and its inverse.
    value.real[value.real == 0] = -0.0
    value.imag[value.imag == 0] = -0.0
    return value


def _as_numeric(exact):
    return QuantumStrategy(_numeric(exact.state), tuple(tuple(tuple(_numeric(e) for e in povm)
                                                            for povm in party) for party in exact.measurements))


def test_numeric_complex_unequal_factor_permutation_and_inverse_preserve_all_bits():
    exact, mapping, game = _fixture()
    source = _as_numeric(exact)
    target = relabel_quantum_strategy(source, mapping)
    assert target.dimensions == (3, 1, 2) and target.tolerance == source.tolerance
    expected = source.state.reshape(2, 3, 1, 2, 3, 1).transpose(1, 2, 0, 4, 5, 3).reshape(6, 6)
    assert target.state.tobytes() == expected.tobytes()
    assert target.score(relabel_game(game, mapping)) == pytest.approx(source.score(game), abs=1e-14)
    original_p, transported_p = source.probabilities(mapping.source), target.probabilities(mapping.target)
    for xi, x in enumerate(mapping.target.inputs):
        for ai, a in enumerate(mapping.target.outputs):
            old = mapping.source.input_index(mapping.source_input(x))*6 + mapping.source.output_index(mapping.source_output(x, a))
            assert transported_p[xi*6+ai] == pytest.approx(original_p[old], abs=1e-15)
    restored = relabel_quantum_strategy(target, mapping.inverse())
    assert restored.state.tobytes() == source.state.tobytes()
    for old_party, restored_party in zip(source.measurements, restored.measurements):
        for old_povm, restored_povm in zip(old_party, restored_party):
            assert all(a.tobytes() == b.tobytes() for a, b in zip(old_povm, restored_povm))


def test_exact_transport_preserves_scores_dimensions_inherited_policy_and_composition():
    source, mapping, game = _fixture()
    target = relabel_exact_strategy(source, mapping)
    assert target.dimensions == (3, 1, 2) and target.state.dimension == 6
    assert target._budget is source._budget
    assert target.score(relabel_game(game, mapping)) == source.score(game) == Fraction(-1, 30)
    assert relabel_exact_strategy(target, mapping.inverse()) == source
    next_map = Relabelling(mapping.target, (2, 0, 1))
    sequential = relabel_exact_strategy(target, next_map)
    assert sequential == relabel_exact_strategy(source, mapping.then(next_map))


def test_matrix_budgets_are_checked_before_basis_permutation(monkeypatch):
    import nonlocalgames.equivalence.quantum as transport
    exact, mapping, _ = _fixture()
    numeric = _as_numeric(exact)
    monkeypatch.setattr(transport, '_basis_indices', lambda *args: pytest.fail('must reject before permutation allocation'))
    for options in ({'max_joint_dimension': 5}, {'max_matrix_entries': 81}):
        with pytest.raises(RelabellingBudgetExceeded):
            relabel_quantum_strategy(numeric, mapping, **options)
    for policy in (replace(exact._budget, max_matrix_entries=81),
                   replace(exact._budget, max_psd_size=11), replace(exact._budget, max_scalar_bits=1)):
        with pytest.raises(ExactBudgetExceeded):
            relabel_exact_strategy(exact, mapping, budget=policy)


@pytest.mark.parametrize('failure_type', [ValueError, np.linalg.LinAlgError])
def test_numerical_revalidation_failure_does_not_widen_tolerance_or_repair(monkeypatch, failure_type):
    source = chsh_strategy()
    original = QuantumStrategy.validate
    seen = []
    def controlled(instance):
        seen.append(instance.tolerance)
        if instance is not source:
            raise failure_type('controlled tolerance-boundary failure')
        return original(instance)
    monkeypatch.setattr(QuantumStrategy, 'validate', controlled)
    with pytest.raises(NumericalRelabellingError, match='unchanged tolerance'):
        relabel_quantum_strategy(source, Relabelling(Scenario((2, 2), (2, 2)), (1, 0)))
    assert seen == [source.tolerance, source.tolerance]


def test_interfaces_and_plain_strategy_types_are_not_silently_coerced():
    exact, mapping, _ = _fixture()
    with pytest.raises(TypeError):
        relabel_quantum_strategy(exact, mapping)
    with pytest.raises(TypeError):
        relabel_exact_strategy(_as_numeric(exact), mapping)
    with pytest.raises(ValueError, match='source interface'):
        relabel_quantum_strategy(_as_numeric(exact), Relabelling(Scenario((1, 1, 1), (2, 3, 1))))
    with pytest.raises(ValueError, match='source interface'):
        relabel_exact_strategy(exact, Relabelling(Scenario((1, 1, 1), (2, 3, 1))))


def test_many_trivial_factors_need_no_high_rank_numpy_reshape_or_behavior_expansion():
    count = 40
    scenario = Scenario((2,)*count, (2,)*count)
    mapping = Relabelling(scenario, tuple(reversed(range(count))))
    half = np.array([[.5]], dtype=complex)
    numeric = QuantumStrategy(np.ones((1, 1)), (((half, half),)*2,)*count)
    assert relabel_quantum_strategy(numeric, mapping).dimensions == (1,)*count
    policy = ExactBudget(max_parties=count, max_events=1)
    exact_half = RationalMatrix(((Fraction(1, 2),),), budget=policy)
    exact = ExactQuantumStrategy(identity(1, budget=policy), (((exact_half, exact_half),)*2,)*count, budget=policy)
    transported = relabel_exact_strategy(exact, mapping)
    assert transported == exact and transported._budget.max_events == 1


def test_exact_transport_does_not_reset_explicit_larger_dimension_limits():
    policy = ExactBudget(max_joint_dimension=20, max_psd_size=40)
    state = scale(identity(20, budget=policy), Fraction(1, 20), budget=policy)
    source = ExactQuantumStrategy(state, (((identity(4, budget=policy),),),
                                          ((identity(5, budget=policy),),)), budget=policy)
    target = relabel_exact_strategy(source, Relabelling(source.scenario, (1, 0)))
    assert target.dimensions == (5, 4) and target._budget is policy


@pytest.mark.parametrize('permutation', [(0, 0), (True, False), (0.0, 1), (0,)])
def test_shared_density_helper_rejects_invalid_party_bijections(permutation):
    with pytest.raises(ValueError):
        _permute_numeric_density(np.eye(6)/6, (2, 3), permutation)


def test_source_eigensolver_failure_is_distinct_and_precedes_state_copy(monkeypatch):
    import nonlocalgames.equivalence.quantum as transport
    source = chsh_strategy()
    def failure(instance):
        raise np.linalg.LinAlgError('controlled source eigensolver failure')
    monkeypatch.setattr(QuantumStrategy, 'validate', failure)
    monkeypatch.setattr(transport, '_permute_numeric_density', lambda *a, **k: pytest.fail('source must validate first'))
    with pytest.raises(NumericalRelabellingError, match='Source numerical'):
        relabel_quantum_strategy(source, Relabelling(Scenario((2, 2), (2, 2))))
