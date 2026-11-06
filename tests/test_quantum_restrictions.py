"""Restricted searches retain their state promises and expose numerical evidence."""
import numpy as np
import pytest

from nonlocalgames.games import chsh, ghz
from nonlocalgames.quantum import (FixedState, chsh_strategy, ghz_strategy, optimize_state,
                                  partial_trace,
                                  seesaw, state_diagnostics)
from nonlocalgames.quantum.restricted import _repair_state


def _pure(vector):
    vector = np.asarray(vector, dtype=complex)
    vector /= np.linalg.norm(vector)
    return np.outer(vector, vector.conj())


def test_fixed_bell_state_search_never_changes_state():
    fixed = FixedState(chsh_strategy().state, (2, 2))
    result = seesaw(chsh(), fixed_state=fixed, seed=7, restarts=2, max_iterations=15,
                     solver_options={"eps": 1e-8})
    assert result.value == pytest.approx((2 + np.sqrt(2))/4, abs=1e-6)
    np.testing.assert_array_equal(result.strategy.state, fixed.state)
    assert result.fixed_state and not result.state_updates
    assert result.guarantee == "numerical_feasible"


def test_fixed_product_state_has_only_local_advantage():
    fixed = FixedState(_pure([1, 0, 0, 0]), (2, 2))
    result = seesaw(chsh(), fixed_state=fixed, seed=7, restarts=2, max_iterations=8,
                     solver_options={"eps": 1e-8})
    assert result.value == pytest.approx(3/4, abs=1e-6)
    np.testing.assert_array_equal(result.strategy.state, fixed.state)
    with pytest.raises(ValueError, match="dimensions"):
        seesaw(chsh(), (2, 3), fixed_state=fixed)
    with pytest.raises(ValueError, match="PPT"):
        seesaw(chsh(), fixed_state=FixedState(chsh_strategy().state, (2, 2)), ppt_cuts=((0,),))


def test_state_block_has_analytic_fixed_measurement_values():
    measurements = chsh_strategy().measurements
    unconstrained = optimize_state(chsh(), measurements)
    ppt = optimize_state(chsh(), measurements, ppt_cuts=((0,),), solver_options={"eps": 1e-8})
    assert unconstrained.value == pytest.approx((2 + np.sqrt(2))/4, abs=1e-12)
    assert ppt.value == pytest.approx(0.5 + np.sqrt(2)/8, abs=1e-7)
    assert ppt.value == ppt.strategy.score(chsh())
    assert ppt.diagnostics["minimum_ppt_eigenvalue"] >= -1e-9
    assert ppt.guarantee == "numerical_feasible"
    assert ppt.diagnostics["objective_guarantee"] == "numerical_block_optimum"
    assert not ppt.diagnostics["certified"]


def test_ppt_search_cannot_recover_qubit_chsh_violation():
    result = seesaw(chsh(), ppt_cuts=((0,),), restarts=2, max_iterations=8, seed=7,
                     solver_options={"eps": 1e-8})
    assert result.value == pytest.approx(3/4, abs=1e-6)
    assert result.ppt_cuts == ((0,),)
    assert result.state_updates
    assert all(update["identity_mixing_weight"] >= 0 for update in result.state_updates)
    state_diagnostics(result.strategy.state, (2, 2), ppt_cuts=((0,),), tolerance=1e-8)


def test_state_update_supports_multipartite_measurements():
    result = optimize_state(ghz(), ghz_strategy().measurements)
    assert result.value == pytest.approx(1, abs=1e-12)
    assert result.strategy.dimensions == (2, 2, 2)
    ppt = optimize_state(ghz(), ghz_strategy().measurements, ppt_cuts=((0,), (1,), (2,)),
                         solver_options={"eps": 1e-8})
    assert ppt.value == pytest.approx(5/8, abs=1e-7)
    state_diagnostics(ppt.strategy.state, (2, 2, 2), ppt_cuts=((0,), (1,), (2,)))


def test_ppt_repair_simultaneously_enforces_state_and_transpose_positivity():
    raw = chsh_strategy().state
    repaired, diagnostics = _repair_state(raw, (2, 2), ((0,),), 1e-8)
    assert diagnostics["identity_mixing_weight"] > 0.6
    state_diagnostics(repaired, (2, 2), tolerance=1e-10, ppt_cuts=((0,),))
    assert np.trace(repaired).real == pytest.approx(1)


def test_resource_guards_before_dense_search():
    with pytest.raises(ValueError, match="max_joint_dimension"):
        seesaw(chsh(), (20, 20))
    with pytest.raises(ValueError, match="max_joint_dimension"):
        optimize_state(chsh(), chsh_strategy().measurements, max_joint_dimension=3)
    with pytest.raises(ValueError, match="indices"):
        partial_trace(np.eye(4)/4, (2, 2), (0, 0))


def test_quantum_strategy_arrays_cannot_be_made_writable():
    strategy = chsh_strategy()
    for matrix in (strategy.state, strategy.measurements[0][0][0]):
        with pytest.raises(ValueError):
            matrix.setflags(write=True)
        with pytest.raises(ValueError):
            matrix[0, 0] = 0


@pytest.mark.parametrize("tolerance", [True, np.bool_(True)])
def test_strategy_rejects_boolean_tolerances(tolerance):
    from nonlocalgames.quantum import QuantumStrategy
    strategy = chsh_strategy()
    with pytest.raises(ValueError, match="tolerance"):
        QuantumStrategy(strategy.state, strategy.measurements, tolerance=tolerance)
