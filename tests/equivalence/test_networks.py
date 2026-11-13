"""Network witness relabelling keeps source factorization and endpoint order."""
from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.networks import ClassicalNetworkStrategy, QuantumNetworkStrategy, QuantumSource, SourceNetwork
from nonlocalgames.equivalence.finite import relabel_game
from nonlocalgames.equivalence.maps import Relabelling, RelabellingBudgetExceeded
from nonlocalgames.equivalence.networks import (relabel_classical_network_strategy,
                                               relabel_quantum_network_strategy)
from nonlocalgames.equivalence.quantum import NumericalRelabellingError, _permute_numeric_density


def _classical():
    graph = SourceNetwork(3, ((2, 0), (1, 2), (0, 1)))
    scenario = Scenario((2, 1, 3), (2, 3, 2))
    laws = ((Fraction(1, 3), Fraction(2, 3)), (Fraction(1, 4), Fraction(3, 4)),
            (Fraction(1, 5), Fraction(4, 5)))
    rules = []
    for party, outputs in enumerate(scenario.output_sizes):
        def rule(x, latent, a, party=party, outputs=outputs):
            weights = [1 + (j + party*x + sum((k+1)*v for k, v in enumerate(latent))) % 4
                       for j in range(outputs)]
            return Fraction(weights[a], sum(weights))
        rules.append(rule)
    return ClassicalNetworkStrategy.from_response_rules(graph, scenario, laws, rules)


def _expected_behavior(old, relabelling):
    width = len(relabelling.source.outputs)
    return tuple(old[relabelling.source.input_index(relabelling.source_input(x))*width
                     + relabelling.source.output_index(relabelling.source_output(x, a))]
                 for x in relabelling.target.inputs for a in relabelling.target.outputs)


def test_classical_conditional_rows_relabel_exactly_with_unequal_alphabets():
    strategy = _classical()
    relabelling = Relabelling(strategy.scenario, (2, 0, 1),
        ((2, 0, 1), (1, 0), (0,)), (((1, 0), (0, 1), (1, 0)), ((0, 1), (1, 0)), ((2, 0, 1),)))
    moved = relabel_classical_network_strategy(strategy, relabelling)
    assert moved.source_probabilities == strategy.source_probabilities
    assert moved.network.party_ports == tuple(strategy.network.party_ports[i] for i in (2, 0, 1))
    assert moved.probabilities() == _expected_behavior(strategy.probabilities(), relabelling)
    weights = {x: Fraction(i+1, 21) for i, x in enumerate(strategy.scenario.inputs)}
    game = Game.from_rule(strategy.scenario, lambda x, a: Fraction(sum((i+1)*v for i, v in enumerate(a))-sum(x), 7),
                          distribution=weights)
    assert moved.score(relabel_game(game, relabelling)) == strategy.score(game)
    assert relabel_classical_network_strategy(moved, relabelling.inverse()) == strategy
    entries = sum(map(len, strategy.source_probabilities)) + sum(map(len, strategy.responses))
    with pytest.raises(RelabellingBudgetExceeded):
        relabel_classical_network_strategy(strategy, relabelling, max_entries=entries-1)


def _projectors(observable):
    unit = np.eye(observable.shape[0])
    return ((unit+observable)/2, (unit-observable)/2)


def _quantum():
    graph = SourceNetwork(3, ((1, 0), (2, 1)))
    first = np.array([1, 0, 1j, 0, 0, 1], dtype=complex)/np.sqrt(3)
    second = np.array([1, 1j, -1, 2j], dtype=complex)/np.sqrt(7)
    sources = (QuantumSource(np.outer(first, first.conj()), (3, 2)),
               QuantumSource(np.outer(second, second.conj()), (2, 2)))
    x, y, z = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
    fourier = np.exp(2j*np.pi*np.outer(np.arange(6), np.arange(6))/6)/np.sqrt(6)
    central = tuple(sum(np.outer(fourier[:, k], fourier[:, k].conj()) for k in (i, i+3)) for i in range(3))
    measurements = ((_projectors(z), _projectors(y)), (central,), (_projectors(x), _projectors(z)))
    return QuantumNetworkStrategy(graph, sources, measurements, tolerance=1e-8)


def test_quantum_transport_preserves_complex_sources_and_local_factor_order():
    strategy = _quantum()
    source = Scenario((2, 1, 2), (2, 3, 2))
    relabelling = Relabelling(source, (1, 2, 0), ((0,), (1, 0), (0, 1)),
        (((2, 0, 1),), ((1, 0), (0, 1)), ((1, 0), (1, 0))))
    moved = relabel_quantum_network_strategy(strategy, relabelling)
    assert moved.network.sources == ((0, 2), (1, 0))
    assert moved.dimensions == (6, 2, 2)
    assert all(new is old for new, old in zip(moved.sources, strategy.sources))
    assert moved.tolerance == strategy.tolerance
    expected_state = _permute_numeric_density(strategy.as_strategy().state, strategy.dimensions, (1, 2, 0))
    np.testing.assert_array_equal(moved.as_strategy().state, expected_state)
    np.testing.assert_allclose(moved.probabilities(relabelling.target),
                               _expected_behavior(strategy.probabilities(source), relabelling), atol=1e-12, rtol=0)
    restored = relabel_quantum_network_strategy(moved, relabelling.inverse())
    assert restored.network == strategy.network
    np.testing.assert_array_equal(restored.as_strategy().state, strategy.as_strategy().state)
    for original_party, restored_party in zip(strategy.measurements, restored.measurements):
        for original_povm, restored_povm in zip(original_party, restored_party):
            for original, restored_effect in zip(original_povm, restored_povm):
                assert original.tobytes() == restored_effect.tobytes()


def test_network_numeric_failure_does_not_retry_or_widen_tolerance(monkeypatch):
    import nonlocalgames.networks.quantum as model
    strategy = _quantum()
    relabelling = Relabelling(Scenario((2, 1, 2), (2, 3, 2)), (1, 2, 0))
    calls = []
    def reject(self):
        calls.append(self.tolerance)
        raise ValueError('controlled target numerical failure')
    monkeypatch.setattr(model.QuantumNetworkStrategy, '__post_init__', reject)
    with pytest.raises(NumericalRelabellingError, match='unchanged tolerance'):
        relabel_quantum_network_strategy(strategy, relabelling)
    assert calls == [1e-8]


def test_network_budgets_and_wrong_interfaces_fail_before_source_validation(monkeypatch):
    strategy = _quantum()
    relabelling = Relabelling(Scenario((2, 1, 2), (2, 3, 2)), (1, 2, 0))
    monkeypatch.setattr(QuantumNetworkStrategy, 'validate', lambda *a, **k: pytest.fail('must preflight first'))
    for options in ({'max_total_dimension': 23}, {'max_matrix_entries': 575}):
        with pytest.raises(RelabellingBudgetExceeded):
            relabel_quantum_network_strategy(strategy, relabelling, **options)
    with pytest.raises(ValueError, match='interface'):
        relabel_quantum_network_strategy(strategy, Relabelling(Scenario((1, 1, 1), (2, 3, 2))))
    with pytest.raises(TypeError, match='provenance'):
        relabel_quantum_network_strategy(strategy.as_strategy(), relabelling)
