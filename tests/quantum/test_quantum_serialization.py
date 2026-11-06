"""Numerical witness persistence retains matrices and rechecks their semantics."""
import copy
import json

import numpy as np
import pytest

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.games import chsh
from nonlocalgames.networks import bilocal_entanglement_swapping
from nonlocalgames.quantum import QuantumStrategy, chsh_strategy
from nonlocalgames.quantum.serialization import strategy_from_dict, strategy_to_dict


def test_complex_strategy_roundtrip_preserves_state_and_measurement_data():
    source = chsh_strategy()
    phase = np.diag([1, 1j])
    global_phase = np.kron(phase, np.eye(2))
    measurements = (tuple(tuple(phase @ e @ phase.conj().T for e in povm)
                           for povm in source.measurements[0]), source.measurements[1])
    strategy = QuantumStrategy(global_phase @ source.state @ global_phase.conj().T, measurements)
    payload = json.loads(json.dumps(strategy_to_dict(strategy), allow_nan=False))
    recovered = strategy_from_dict(payload)
    np.testing.assert_array_equal(recovered.state, strategy.state)
    for old_party, new_party in zip(strategy.measurements, recovered.measurements):
        for old_povm, new_povm in zip(old_party, new_party):
            for old, new in zip(old_povm, new_povm):
                np.testing.assert_array_equal(old, new)
    assert recovered.dimensions == (2, 2)
    task = AnalysisTask(chsh(), ResourceSpec("quantum_finite", (2, 2), measurement_class="projective"))
    assert task.evaluate(recovered).value == pytest.approx((2 + np.sqrt(2))/4)
    with pytest.raises(ValueError):
        recovered.state.setflags(write=True)


def test_serialization_cannot_strip_network_provenance_implicitly():
    network_strategy = bilocal_entanglement_swapping()
    with pytest.raises(TypeError, match="provenance"):
        strategy_to_dict(network_strategy)


@pytest.mark.parametrize("damage", [
    lambda data: data.update(schema="unknown"),
    lambda data: data.update(encoding="exact"),
    lambda data: data.update(extra=True),
    lambda data: data.update(dimensions=[1, 4]),
    lambda data: data.update(tolerance=True),
    lambda data: data["state"].update(imag=[[0]]),
    lambda data: data["state"]["real"][0].__setitem__(0, float("nan")),
    lambda data: data["state"]["real"][0].__setitem__(0, "0.5"),
    lambda data: data["state"]["real"][0].__setitem__(0, -1),
    lambda data: data["measurements"][0][0][0]["real"][0].__setitem__(0, 0),
])
def test_deserialization_rejects_malformed_or_nonphysical_data(damage):
    payload = copy.deepcopy(strategy_to_dict(chsh_strategy()))
    damage(payload)
    with pytest.raises((ValueError, TypeError)):
        strategy_from_dict(payload)


def test_deserialization_honors_allocation_guards():
    payload = strategy_to_dict(chsh_strategy())
    with pytest.raises(ValueError, match="max_joint_dimension"):
        strategy_from_dict(payload, max_joint_dimension=3)
    with pytest.raises(ValueError, match="max_matrix_entries"):
        strategy_from_dict(payload, max_matrix_entries=47)
