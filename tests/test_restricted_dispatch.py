"""Task dispatch must preserve restrictions rather than choose a nearby resource model."""
import numpy as np
import pytest

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.games import chsh
from nonlocalgames.quantum import FixedState, chsh_strategy


def test_task_search_propagates_fixed_state_and_actual_dimensions():
    fixed = FixedState(chsh_strategy().state, (2, 2))
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', (3, 3), fixed_state=fixed))
    result = task.solve(method='seesaw', restarts=1, max_iterations=10, seed=7,
                        solver_options={'eps': 1e-8})
    assert result.value == pytest.approx((2 + np.sqrt(2))/4, abs=1e-6)
    np.testing.assert_array_equal(result.witness.state, fixed.state)
    assert result.witness.dimensions == (2, 2)
    assert result.guarantee == 'numerical_witness'
    with pytest.raises(ValueError, match='ResourceSpec'):
        task.solve(method='seesaw', dimensions=(4, 4))
    with pytest.raises(NotImplementedError):
        task.solve(method='npa')


def test_task_ppt_state_block_is_not_a_global_upper_bound():
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', (2, 2),
                        state_constraint='ppt', ppt_cuts=((0,),), measurement_class='projective'))
    result = task.solve(method='state_sdp', measurements=chsh_strategy().measurements,
                        solver_options={'eps': 1e-8})
    assert result.value == pytest.approx(.5 + np.sqrt(2)/8, abs=1e-7)
    assert result.guarantee == 'numerical_witness'
    assert 'fixed supplied measurements' in result.diagnostics['scope']


def test_invalid_state_block_measurements_fail_before_optimizer(monkeypatch):
    import nonlocalgames.quantum
    monkeypatch.setattr(nonlocalgames.quantum, 'optimize_state',
                        lambda *args, **kwargs: pytest.fail('optimizer must not run'))
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', (1, 1)))
    with pytest.raises(ValueError, match='dimension bounds'):
        task.solve(method='state_sdp', measurements=chsh_strategy().measurements)
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', measurement_class='projective'))
    measurements = tuple(tuple((np.eye(2)/2, np.eye(2)/2) for _ in range(2)) for _ in range(2))
    with pytest.raises(ValueError, match='projective'):
        task.solve(method='state_sdp', measurements=measurements)


def test_fixed_state_cannot_be_replaced_by_state_optimization():
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite',
                        fixed_state=FixedState(np.eye(4)/4, (2, 2))))
    with pytest.raises(NotImplementedError):
        task.solve(method='state_sdp', measurements=chsh_strategy().measurements)


def test_task_ppt_search_retains_transpose_constraint():
    task = AnalysisTask(chsh(), ResourceSpec('quantum_finite', (2, 2),
                        state_constraint='ppt', ppt_cuts=((0,),)))
    result = task.solve(method='seesaw', seed=7, restarts=1, max_iterations=8,
                        solver_options={'eps': 1e-8})
    assert result.value == pytest.approx(.75, abs=1e-6)
    assert result.diagnostics['minimum_ppt_eigenvalue'] >= -1e-7
