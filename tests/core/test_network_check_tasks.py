from fractions import Fraction

import pytest

from nonlocalgames import NetworkCompatibilityTask, ResourceSpec, Scenario
from nonlocalgames.core.compatibility_serialization import (compatibility_from_dict,
    compatibility_to_dict, dump_compatibility, load_compatibility)
from nonlocalgames.networks import RationalBehavior, SourceNetwork


def w_task(kind='local', inputs=(1, 1, 1)):
    s = Scenario(inputs, (2, 2, 2))
    data = RationalBehavior(s, tuple(Fraction(1, 3) if sum(a) == 1 else 0
                                      for x in s.inputs for a in s.outputs))
    r = ResourceSpec(kind, network=SourceNetwork(3, ((0, 1), (1, 2), (2, 0))),
                     shared_randomness='forbidden')
    return NetworkCompatibilityTask(data, r)


def test_spiral_scope_cannot_be_upgraded_to_quantum_or_multiple_inputs():
    for task in (w_task('quantum_finite'), w_task(inputs=(2, 1, 1))):
        assert 'spiral' not in task.capabilities()
        with pytest.raises(NotImplementedError):
            task.check(method='spiral')
        assert task.check(method='triangle_finner').status == 'not_ruled_out'


def test_classical_spiral_detects_more_than_finner():
    task = w_task()
    assert task.check(method='triangle_finner').status == 'not_ruled_out'
    result = task.check(method='spiral')
    assert result.status == 'incompatible'
    assert result.guarantee == 'exact_classical_incompatibility'
    assert result.resource == 'classical_fanout'


def test_task_serialization_and_applicability_guards(tmp_path):
    task = w_task()
    assert compatibility_from_dict(compatibility_to_dict(task)) == task
    path = tmp_path / 'check.json'
    dump_compatibility(task, path)
    assert load_compatibility(path) == task
    with pytest.raises(ValueError, match='max_file_bytes'):
        load_compatibility(path, max_file_bytes=4)
    with pytest.raises(ValueError):
        NetworkCompatibilityTask(task.behavior, ResourceSpec('local'))
    with pytest.raises(NotImplementedError):
        task.check(method='bilocal_chain')


def test_exact_bilocal_membership_remains_classical_complete_fragment():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    behavior = RationalBehavior(scenario, (Fraction(1, 8),)*8)
    task = NetworkCompatibilityTask(behavior, ResourceSpec('local',
        network=SourceNetwork(3, ((0, 1), (1, 2))), shared_randomness='forbidden'))
    result = task.check(method='bilocal_chain')
    assert result.status == 'compatible' and result.guarantee == 'exact_membership'
    assert result.witness.probabilities() == behavior.probabilities
