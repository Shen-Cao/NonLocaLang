"""Decoders enforce numerical data contracts before expensive matrix work."""
import copy

import numpy as np
import pytest

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.core.resource_serialization import (dump_task, load_task, resource_from_dict,
                                                       resource_to_dict)
from nonlocalgames.games import chsh
from nonlocalgames.networks import bilocal_entanglement_swapping
from nonlocalgames.networks.serialization import quantum_from_dict, quantum_to_dict
from nonlocalgames.quantum import FixedState, chsh_strategy
from nonlocalgames.quantum.states import state_diagnostics


def fixed_payload():
    return resource_to_dict(ResourceSpec('quantum_finite', fixed_state=FixedState(chsh_strategy().state, (2, 2))))


@pytest.mark.parametrize('bad', ['0.5', True, float('inf'), float('nan'), 10**1000])
def test_fixed_and_network_loaders_reject_non_float64_entries(bad):
    fixed = fixed_payload()
    fixed['fixed_state']['real'][0][0] = bad
    with pytest.raises(ValueError):
        resource_from_dict(fixed)
    network = quantum_to_dict(bilocal_entanglement_swapping())
    network['sources'][0]['state']['real'][0][0] = bad
    with pytest.raises(ValueError):
        quantum_from_dict(network)


def test_resource_dimension_and_entry_guards_precede_decoding(monkeypatch):
    import nonlocalgames.core.resource_serialization as module
    monkeypatch.setattr(module, 'matrix_from_dict', lambda *args: pytest.fail('must check budget first'))
    for options in ({'max_joint_dimension': 3}, {'max_matrix_entries': 15}):
        with pytest.raises(ValueError, match='exceeds max_'):
            resource_from_dict(fixed_payload(), **options)


def test_network_total_dimension_guard_precedes_source_validation(monkeypatch):
    payload = quantum_to_dict(bilocal_entanglement_swapping())
    import nonlocalgames.networks.serialization as module
    monkeypatch.setattr(module, 'matrix_from_dict', lambda *args: pytest.fail('must check budget first'))
    for options in ({'max_total_dimension': 15}, {'max_matrix_entries': 1}):
        with pytest.raises(ValueError, match='exceeds? max_'):
            quantum_from_dict(payload, **options)


def test_task_file_and_matrix_guards_are_configurable(tmp_path):
    task = AnalysisTask(chsh(), resource_from_dict(fixed_payload()))
    path = tmp_path / 'task.json'
    dump_task(task, path)
    with pytest.raises(ValueError, match='max_file_bytes'):
        load_task(path, max_file_bytes=10)
    with pytest.raises(ValueError, match='max_joint_dimension'):
        load_task(path, max_joint_dimension=3)
    assert load_task(path, max_file_bytes=100_000).game == task.game


@pytest.mark.parametrize('bad', [True, np.bool_(True), False, '1e-9', complex(1, 0)])
def test_state_tolerance_rejects_boolean_and_nonreal_values(bad):
    with pytest.raises(ValueError, match='tolerance'):
        state_diagnostics(np.eye(2)/2, (2,), tolerance=bad)


def test_network_shape_checked_before_eigendecomposition():
    payload = quantum_to_dict(bilocal_entanglement_swapping())
    broken = copy.deepcopy(payload)
    broken['sources'][0]['dimensions'] = [2]
    with pytest.raises(ValueError, match='endpoint'):
        quantum_from_dict(broken)
    broken = copy.deepcopy(payload)
    broken['measurements'][0][0][0]['real'] = [[1]]
    with pytest.raises(ValueError, match='dimension'):
        quantum_from_dict(broken)
