"""State specifications retain tensor-factor order and exact numerical input data."""
import numpy as np
import pytest

from nonlocalgames.quantum.strategy import chsh_strategy
from nonlocalgames.quantum.states import (FixedState, partial_trace, partial_transpose,
                                         product_state_residual, state_diagnostics,
                                         validate_ppt_cuts)


def _pure(vector):
    vector = np.asarray(vector, dtype=complex)
    vector /= np.linalg.norm(vector)
    return np.outer(vector, vector.conj())


def test_tensor_operations_with_complex_unequal_factors_and_order():
    local = (_pure([1, 1j]), _pure([1, 2j, -1]), _pure([1, -2]))
    state = np.kron(np.kron(local[0], local[1]), local[2])
    np.testing.assert_allclose(partial_trace(state, (2, 3, 2), (2, 0)),
                               np.kron(local[2], local[0]), atol=1e-14)
    np.testing.assert_allclose(partial_trace(state, (2, 3, 2), ()), [[1]], atol=1e-14)
    np.testing.assert_allclose(partial_transpose(state, (2, 3, 2), (0, 2)),
                               np.kron(np.kron(local[0].T, local[1]), local[2].T), atol=1e-14)
    assert product_state_residual(state, (2, 3, 2)) < 1e-14
    assert product_state_residual(chsh_strategy().state, (2, 2)) > 0.5


def test_fixed_state_is_validated_and_immutable():
    state = np.eye(6) / 6
    fixed = FixedState(state, (2, 3))
    state[0, 0] = 0
    assert fixed.state[0, 0] == 1/6
    with pytest.raises(ValueError):
        fixed.state.setflags(write=True)
    with pytest.raises(ValueError):
        FixedState(np.eye(4), (2, 2))
    with pytest.raises(ValueError):
        FixedState(np.diag([-0.1, 1.1]), (2,))
    with pytest.raises(ValueError):
        FixedState(np.eye(4)/4, (2, 3))
    with pytest.raises(ValueError):
        FixedState(np.eye(4)/4, (2, 2), tolerance=np.nan)


def test_ppt_validation_and_cut_canonicalization():
    assert validate_ppt_cuts(((1,),), 2) == ((0,),)
    for cuts in (((),), ((0, 1),), ((2,),), ((0,), (1,))):
        with pytest.raises(ValueError):
            validate_ppt_cuts(cuts, 2)
    with pytest.raises(ValueError, match="PPT"):
        state_diagnostics(chsh_strategy().state, (2, 2), ppt_cuts=((0,),))
