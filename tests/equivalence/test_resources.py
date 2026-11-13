"""Relabelling transports the full resource contract without numerical repair."""
import numpy as np
import pytest

from nonlocalgames import AnalysisTask, ResourceSpec, Scenario
from nonlocalgames.core.compatibility_tasks import NetworkCompatibilityTask
from nonlocalgames.core.identity import resource_id
from nonlocalgames.games import chsh
from nonlocalgames.networks import RationalBehavior, SourceNetwork
from nonlocalgames.quantum.states import FixedState, validate_ppt_cuts
from nonlocalgames.equivalence.maps import Relabelling, RelabellingBudgetExceeded
from nonlocalgames.equivalence.quantum import NumericalRelabellingError
from nonlocalgames.equivalence.resources import relabel_resource, relabel_task


def test_symbolic_dimensions_ppt_cuts_and_policies_are_transported():
    source = Scenario((2, 3, 1), (2, 4, 2))
    relabelling = Relabelling(source, (2, 0, 1))
    resource = ResourceSpec('quantum_finite', (2000, 5000, 7000), state_constraint='ppt',
                            ppt_cuts=((0,), (0, 1)), measurement_class='projective', shared_randomness='forbidden')
    moved = relabel_resource(resource, relabelling, max_joint_dimension=1, max_matrix_entries=1)
    assert moved.local_dimensions == (7000, 2000, 5000)
    assert moved.ppt_cuts == validate_ppt_cuts(((1,), (1, 2)), 3)
    assert moved.measurement_class == resource.measurement_class
    assert moved.shared_randomness == resource.shared_randomness
    assert relabel_resource(moved, relabelling.inverse()) == resource
    raw = ResourceSpec('quantum_finite', state_constraint='ppt', ppt_cuts=((1,),))
    swap = Relabelling(Scenario((2, 3), (2, 2)), (1, 0))
    assert resource_id(relabel_resource(relabel_resource(raw, swap), swap.inverse())) == resource_id(raw)


def test_network_source_and_endpoint_positions_are_preserved():
    graph = SourceNetwork(3, ((2, 0), (0, 1), (2, 1), (2, 0), (1,)))
    source = Scenario((1, 2, 3), (2, 2, 2))
    relabelling = Relabelling(source, (2, 0, 1))
    resource = ResourceSpec('quantum_finite', (2, 3, 4), network=graph, shared_randomness='forbidden')
    moved = relabel_resource(resource, relabelling)
    assert moved.network.sources == ((0, 1), (1, 2), (0, 2), (0, 1), (2,))
    assert moved.network.party_ports == tuple(graph.party_ports[i] for i in (2, 0, 1))
    assert relabel_resource(moved, relabelling.inverse()) == resource


def _fixed_resource():
    rho_a = np.array([[.5, -.5j], [.5j, .5]], dtype=complex)
    rho_b = np.diag([0., 1., 0.]).astype(complex)
    matrix = np.kron(rho_a, rho_b)
    matrix.real[0, 0] = -0.0
    matrix.imag[2, 2] = -0.0
    return ResourceSpec('quantum_finite', (3, 5), fixed_state=FixedState(matrix, (2, 3), 1e-8),
                        state_constraint='product', measurement_class='projective')


def test_fixed_state_moves_tensor_factors_and_preserves_signed_zero_bytes():
    resource = _fixed_resource()
    relabelling = Relabelling(Scenario((2, 3), (2, 3)), (1, 0))
    moved = relabel_resource(resource, relabelling)
    assert moved.local_dimensions == (5, 3)
    assert moved.fixed_state.dimensions == (3, 2)
    expected = resource.fixed_state.state.reshape(2, 3, 2, 3).transpose(1, 0, 3, 2).reshape(6, 6)
    assert moved.fixed_state.state.tobytes() == expected.tobytes()
    assert moved.fixed_state.tolerance == resource.fixed_state.tolerance
    restored = relabel_resource(moved, relabelling.inverse())
    assert restored.fixed_state.state.tobytes() == resource.fixed_state.state.tobytes()
    assert resource_id(restored) == resource_id(resource)


def test_failed_numeric_revalidation_keeps_original_tolerance_and_data(monkeypatch):
    import nonlocalgames.equivalence.resources as transport
    resource = _fixed_resource()
    original = resource.fixed_state.state.tobytes()
    relabelling = Relabelling(Scenario((2, 3), (2, 3)), (1, 0))
    calls = []
    def reject(state, dimensions, tolerance):
        calls.append((dimensions, tolerance))
        raise ValueError('controlled target numerical rejection')
    monkeypatch.setattr(transport, 'FixedState', reject)
    with pytest.raises(NumericalRelabellingError, match='unchanged tolerance'):
        relabel_resource(resource, relabelling)
    assert calls == [((3, 2), 1e-8)]
    assert resource.fixed_state.state.tobytes() == original


def test_resource_budgets_and_source_counts_are_checked_before_validation(monkeypatch):
    resource = _fixed_resource()
    relabelling = Relabelling(Scenario((2, 3), (2, 3)), (1, 0))
    monkeypatch.setattr(FixedState, 'validate', lambda *a, **k: pytest.fail('budget must be checked first'))
    for limits in ({'max_joint_dimension': 5}, {'max_matrix_entries': 35}):
        with pytest.raises(RelabellingBudgetExceeded):
            relabel_resource(resource, relabelling, **limits)
    with pytest.raises(ValueError, match='party count'):
        relabel_resource(ResourceSpec('quantum_finite', (2,)), relabelling)
    with pytest.raises(ValueError, match='positive integer'):
        relabel_resource(ResourceSpec('local'), relabelling, max_matrix_entries=True)


def test_value_and_compatibility_tasks_keep_transport_scope():
    game = chsh()
    value_task = AnalysisTask(game, ResourceSpec('quantum_finite', (2, 3)))
    moved = relabel_task(value_task, Relabelling(game.scenario, (1, 0)))
    assert moved.resource.local_dimensions == (3, 2)
    assert moved.game.scenario == game.scenario
    with pytest.raises(RelabellingBudgetExceeded):
        relabel_task(value_task, Relabelling(game.scenario), max_entries=15)
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    from fractions import Fraction
    behavior = RationalBehavior(scenario, (Fraction(1, 8),)*8)
    resource = ResourceSpec('local', network=SourceNetwork(3, ((0, 1), (1, 2))), shared_randomness='forbidden')
    task = NetworkCompatibilityTask(behavior, resource)
    assert 'bilocal_chain' in task.capabilities()
    moved = relabel_task(task, Relabelling(scenario, (1, 0, 2)))
    assert moved.resource.network.sources == ((1, 0), (0, 2))
    assert 'bilocal_chain' not in moved.capabilities()
    assert moved.behavior == behavior
