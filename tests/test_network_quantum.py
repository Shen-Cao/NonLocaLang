from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import (BILOCAL_SCENARIO, QuantumNetworkStrategy, QuantumSource,
                                   SourceNetwork, bilocal_deterministic_branches,
                                   bilocal_entanglement_swapping, bilocal_witness)


def test_entanglement_swapping_violation_and_product_sources():
    strategy = bilocal_entanglement_swapping()
    assert strategy.dimensions == (2, 4, 2)
    behavior = strategy.probabilities(BILOCAL_SCENARIO)
    witness = bilocal_witness(behavior)
    assert witness.I == pytest.approx(0.5)
    assert witness.J == pytest.approx(0.5)
    assert witness.value == pytest.approx(np.sqrt(2))
    assert witness.violates_bound and witness.guarantee == "numerical_witness"
    expected = np.kron(strategy.sources[0].state, strategy.sources[1].state)
    np.testing.assert_allclose(strategy.as_strategy().state, expected)
    assert all(abs(np.trace(e) - 1) < 1e-9 for e in strategy.measurements[1][0])


def test_nonlinear_witness_exact_and_global_randomness_counterexample():
    branches = bilocal_deterministic_branches()
    assert [(bilocal_witness(s.probabilities()).I, bilocal_witness(s.probabilities()).J)
            for s in branches] == [(1, 0), (0, 1)]
    assert all(not bilocal_witness(s.probabilities()).violates_bound for s in branches)
    mixture = tuple((a+b)/2 for a, b in zip(*(s.probabilities() for s in branches)))
    witness = bilocal_witness(mixture)
    assert witness.I == witness.J == Fraction(1, 2)
    assert witness.violates_bound and witness.guarantee == "exact_rational_witness"
    assert witness.value == pytest.approx(np.sqrt(2))


def test_source_permutation_with_complex_unequal_endpoint_factors():
    graph = SourceNetwork(3, ((2, 0), (1, 2)))
    # Original factors: party2 dim2, party0 dim3, party1 dim2, party2 dim3.
    vectors = [np.array([1, 1j])/np.sqrt(2), np.array([0, 1, 0]),
               np.array([0, 1]), np.array([1, 0, 1j])/np.sqrt(2)]
    psi0, psi1 = np.kron(*vectors[:2]), np.kron(*vectors[2:])
    sources = (QuantumSource(np.outer(psi0, psi0.conj()), (2, 3)),
               QuantumSource(np.outer(psi1, psi1.conj()), (2, 3)))
    dims = (3, 2, 6)
    strategy = QuantumNetworkStrategy(graph, sources, tuple(((np.eye(d),),) for d in dims))
    expected = np.kron(np.kron(vectors[1], vectors[2]), np.kron(vectors[0], vectors[3]))
    np.testing.assert_allclose(strategy.as_strategy().state, np.outer(expected, expected.conj()))
    np.testing.assert_allclose(strategy.probabilities(Scenario((1, 1, 1), (1, 1, 1))), [1])


def test_isolated_quantum_parties_and_dimension_guards():
    strategy = QuantumNetworkStrategy(SourceNetwork(2, ()), (),
                                      ((((np.array([[0.25]])), np.array([[0.75]])),),
                                       ((np.ones((1, 1)),),)))
    np.testing.assert_allclose(strategy.probabilities(Scenario((1, 1), (2, 1))), [0.25, 0.75])
    source = QuantumSource(np.eye(4)/4, (2, 2))
    graph = SourceNetwork(2, ((0, 1),))
    with pytest.raises(ValueError):
        QuantumNetworkStrategy(graph, (source,), (((np.eye(2),),), ((np.eye(2),),)), max_total_dimension=3)
    with pytest.raises(ValueError):
        QuantumNetworkStrategy(graph, (source,), (((np.eye(4),),), ((np.eye(1),),)))
    with pytest.raises(ValueError):
        QuantumNetworkStrategy(graph, (QuantumSource(np.eye(4)/4, (4,)),), (((np.eye(2),),),)*2)
    with pytest.raises(ValueError):
        QuantumSource(np.diag([2, -1]), (2,))


def test_witness_rejects_malformed_or_complex_data():
    values = list(bilocal_deterministic_branches()[0].probabilities())
    with pytest.raises(ValueError):
        bilocal_witness(values[:-1])
    values[0] = -1
    with pytest.raises(ValueError):
        bilocal_witness(values)
    values = bilocal_entanglement_swapping().probabilities(BILOCAL_SCENARIO).astype(complex)
    with pytest.raises(TypeError):
        bilocal_witness(values)
    with pytest.raises(ValueError):
        bilocal_witness([np.nan]*64)


def test_network_revalidates_source_tolerance_and_protects_factorization():
    source = QuantumSource(np.diag([1.000001, -0.000001]), (2,), tolerance=1e-5)
    graph = SourceNetwork(1, ((0,),))
    with pytest.raises(ValueError):
        QuantumNetworkStrategy(graph, (source,), (((np.eye(2),),),), tolerance=1e-9)
    with pytest.raises(ValueError):
        source.state.setflags(write=True)
    strategy = bilocal_entanglement_swapping()
    with pytest.raises(ValueError):
        strategy.as_strategy().state.setflags(write=True)
    with pytest.raises(ValueError):
        QuantumNetworkStrategy(SourceNetwork(1, ()), (), (((np.ones((1, 1)),),),), tolerance=True)
