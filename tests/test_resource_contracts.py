from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames import (AnalysisTask, MeasurementClass, Resource, ResourceSpec,
                           SharedRandomness)
from nonlocalgames.core.resource_serialization import (dump_task, load_task, resource_from_dict,
                                                       resource_to_dict)
from nonlocalgames.games import chsh, ghz
from nonlocalgames.quantum import QuantumStrategy, chsh_strategy
from nonlocalgames.quantum.states import FixedState


def test_external_randomness_changes_contract_not_local_value():
    for policy in SharedRandomness:
        task = AnalysisTask(chsh(), ResourceSpec(Resource.LOCAL, shared_randomness=policy))
        result = task.solve()
        assert result.value == Fraction(3, 4)
        assert result.diagnostics['shared_randomness'] == policy.value
    assert result.resource == 'local_product'


def test_product_state_resource_has_exact_classical_value():
    for measurements in MeasurementClass:
        r = ResourceSpec(Resource.QUANTUM_FINITE, (1, 3, 2), state_constraint='product',
                         measurement_class=measurements, shared_randomness='forbidden')
        result = AnalysisTask(ghz(), r).solve()
        assert result.value == Fraction(3, 4) and result.guarantee == 'exact'
        assert result.resource == 'quantum_product'


def test_contract_validation_and_capabilities():
    with pytest.raises(ValueError):
        ResourceSpec(Resource.LOCAL, state_constraint='product')
    with pytest.raises(ValueError):
        ResourceSpec(Resource.QUANTUM_FINITE, (2, 2), state_constraint='ppt')
    with pytest.raises(ValueError):
        ResourceSpec(Resource.QUANTUM_FINITE, (2, 2), state_constraint='ppt', ppt_cuts=((0,), (1,)))
    with pytest.raises(ValueError):
        AnalysisTask(chsh(), ResourceSpec(Resource.QUANTUM_FINITE,
                     state_constraint='ppt', ppt_cuts=((0, 1),)))
    r = ResourceSpec(Resource.QUANTUM_FINITE, (2, 2), measurement_class='projective')
    task = AnalysisTask(chsh(), r)
    assert 'seesaw' not in task.capabilities()
    for method in ('npa', 'xor', 'seesaw'):
        with pytest.raises(NotImplementedError):
            task.solve(method=method)


def test_fixed_state_payload_roundtrip_and_immutable(tmp_path):
    original = chsh_strategy()
    fixed = FixedState(original.state, (2, 2))
    task = AnalysisTask(chsh(), ResourceSpec(Resource.QUANTUM_FINITE, (3, 4), fixed_state=fixed))
    path = tmp_path / 'task.json'
    dump_task(task, path)
    restored = load_task(path)
    assert restored.game == task.game
    assert restored.resource.local_dimensions == (3, 4)
    np.testing.assert_array_equal(restored.resource.fixed_state.state, fixed.state)
    with pytest.raises(ValueError):
        fixed.state.setflags(write=True)
    payload = resource_to_dict(task.resource)
    payload['extra'] = 'unexpected'
    with pytest.raises(ValueError):
        resource_from_dict(payload)


def test_state_restrictions_are_enforced_on_concrete_witnesses():
    strategy = chsh_strategy()
    for resource in (
        ResourceSpec(Resource.QUANTUM_FINITE, (1, 2)),
        ResourceSpec(Resource.QUANTUM_FINITE, state_constraint='product'),
        ResourceSpec(Resource.QUANTUM_FINITE, state_constraint='ppt', ppt_cuts=((0,),)),
        ResourceSpec(Resource.QUANTUM_FINITE, fixed_state=FixedState(np.eye(4)/4, (2, 2))),
    ):
        with pytest.raises(ValueError):
            AnalysisTask(chsh(), resource).evaluate(strategy)
    good = ResourceSpec(Resource.QUANTUM_FINITE, (3, 3), measurement_class='projective')
    result = AnalysisTask(chsh(), good).evaluate(strategy)
    assert result.value > .85 and result.guarantee == 'numerical_witness'


def test_projective_contract_rejects_unsharp_povms():
    strategy = chsh_strategy()
    measurements = tuple(tuple(tuple(.9*e + .05*np.eye(2) for e in povm)
                               for povm in party) for party in strategy.measurements)
    noisy = QuantumStrategy(strategy.state, measurements)
    task = AnalysisTask(chsh(), ResourceSpec(Resource.QUANTUM_FINITE, measurement_class='projective'))
    with pytest.raises(ValueError, match='projective'):
        task.evaluate(noisy)


def test_numerical_product_fixed_state_does_not_claim_exact_value():
    task = AnalysisTask(chsh(), ResourceSpec(Resource.QUANTUM_FINITE,
                       fixed_state=FixedState(np.eye(4)/4, (2, 2)), state_constraint='product'))
    assert 'local' not in task.capabilities()
    with pytest.raises(NotImplementedError):
        task.solve()
