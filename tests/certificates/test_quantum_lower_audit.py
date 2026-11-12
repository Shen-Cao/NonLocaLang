"""Independent exact checks of complex tensor algebra, repair, and lower certificates."""
from fractions import Fraction

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.games import chsh, ghz
from nonlocalgames.quantum.exact import (ExactBudget, ExactBudgetExceeded, ExactPSDViolation,
    ExactQuantumStrategy, RationalMatrix, realification, verify_psd_complex)
from nonlocalgames.quantum.exact.algebra import (add, adjoint, identity, kron, multiply,
                                                scale, subtract, trace_product)


def _y_plus():
    half = Fraction(1, 2)
    return RationalMatrix(((half, 0), (0, half)), ((0, -half), (half, 0)))


def _basis(dimension, index):
    return RationalMatrix(tuple(tuple(int(i == j == index) for j in range(dimension))
                                for i in range(dimension)))


def test_realification_is_global_psd_check_not_new_physical_tensor_factors():
    rho_a, rho_b = _y_plus(), _basis(3, 1)
    state = kron(rho_a, rho_b)
    expected_real = [[Fraction(0) for _ in range(6)] for _ in range(6)]
    expected_imag = [[Fraction(0) for _ in range(6)] for _ in range(6)]
    expected_real[1][1] = expected_real[4][4] = Fraction(1, 2)
    expected_imag[1][4], expected_imag[4][1] = -Fraction(1, 2), Fraction(1, 2)
    assert state == RationalMatrix(expected_real, expected_imag)
    assert state.dimension == 6 and len(realification(state)) == 12
    assert verify_psd_complex(state).rank == 2
    povm_a = (rho_a, subtract(identity(2), rho_a))
    povm_b = tuple(_basis(3, i) for i in range(3))
    strategy = ExactQuantumStrategy(state, ((povm_a,), (povm_b,)), dimensions=(2, 3))
    assert strategy.probabilities() == (0, 1, 0, 0, 0, 0)
    assert strategy.validate()['state_rank'] == 1
    assert strategy.validate()['realification_state_rank'] == 2


def test_complex_trace_and_adjoint_signs_are_not_discarded():
    left = RationalMatrix(((0, 0), (0, 0)), ((0, 1), (0, 0)))
    right = RationalMatrix(((0, 0), (1, 0)))
    assert trace_product(left, right) == (0, 1)
    assert multiply(left, right).trace() == (0, 1)
    general = RationalMatrix(((1, 2), (3, 4)), ((0, 1), (-2, 0)))
    positive = multiply(adjoint(general), general)
    assert positive.is_hermitian()
    assert verify_psd_complex(positive).rank == 4
    assert adjoint(adjoint(general)) == general


def test_exact_psd_rejects_tiny_complex_negative_direction_and_nonhermiticity():
    epsilon = Fraction(1, 10**100)
    indefinite = RationalMatrix(((1, 0), (0, 0)), ((0, epsilon), (-epsilon, 0)))
    with pytest.raises(ExactPSDViolation):
        verify_psd_complex(indefinite)
    nonhermitian = RationalMatrix(((1, 0), (0, 1)), ((epsilon, 0), (0, 0)))
    with pytest.raises(ValueError, match='Hermitian'):
        verify_psd_complex(nonhermitian)
    with pytest.raises(ExactBudgetExceeded):
        verify_psd_complex(_y_plus(), budget=ExactBudget(max_psd_size=3))


def _ghz_exact():
    real = [[Fraction(0) for _ in range(8)] for _ in range(8)]
    for i in (0, 7):
        for j in (0, 7):
            real[i][j] = Fraction(1, 2)
    x = RationalMatrix(((0, 1), (1, 0)))
    y = RationalMatrix(((0, 0), (0, 0)), ((0, -1), (1, 0)))
    unit = identity(2)
    party = tuple((scale(add(unit, observable), Fraction(1, 2)),
                   scale(subtract(unit, observable), Fraction(1, 2))) for observable in (x, y))
    return ExactQuantumStrategy(RationalMatrix(real), (party,)*3)


def test_ghz_imaginary_projectors_and_global_conjugation_preserve_exact_behavior():
    strategy = _ghz_exact()
    assert strategy.score(ghz()) == 1
    def conjugate(matrix):
        return RationalMatrix(matrix.real, tuple(tuple(-v for v in row) for row in matrix.imag))
    conjugated = ExactQuantumStrategy(conjugate(strategy.state), tuple(tuple(tuple(conjugate(e)
        for e in povm) for povm in party) for party in strategy.measurements))
    assert conjugated.probabilities() == strategy.probabilities()
    probabilities = strategy.probabilities()
    x = (0, 1, 1)
    start = strategy.scenario.input_index(x) * 8
    assert all(probabilities[start+i] == (Fraction(1, 4) if sum(a) % 2 else 0)
               for i, a in enumerate(strategy.scenario.outputs))


def test_lower_certificate_handles_signed_scores_without_numerical_proof_tools(monkeypatch):
    from nonlocalgames.certificates.quantum import certify_quantum_strategy, verify_quantum_lower_certificate
    import cvxpy as cp
    import numpy as np
    rho_a, rho_b = _y_plus(), _basis(3, 1)
    strategy = ExactQuantumStrategy(kron(rho_a, rho_b),
        (((rho_a, subtract(identity(2), rho_a)),), (tuple(_basis(3, i) for i in range(3)),)))
    game = Game.from_rule(Scenario((1, 1), (2, 3)), lambda x, a: Fraction(-7, 3) if a == (0, 1) else -5)
    def forbidden(*args, **kwargs):
        pytest.fail('exact verification must not call numerical optimization or eigenvalue routines')
    for name in ('eigvalsh', 'eigh', 'eigvals', 'cholesky'):
        monkeypatch.setattr(np.linalg, name, forbidden)
    monkeypatch.setattr(cp.Problem, 'solve', forbidden)
    result = certify_quantum_strategy(game, strategy)
    assert result.value == Fraction(-7, 3) and result.guarantee == 'certified_lower_bound'
    assert verify_quantum_lower_certificate(game, result.witness).value == result.value


def test_zero_step_repair_certifies_changed_uniform_povm_not_original_score():
    from nonlocalgames.quantum import chsh_strategy
    from nonlocalgames.quantum.exact.rounding import rationalize_strategy
    original = chsh_strategy()
    result = rationalize_strategy(original, max_denominator=3, repair_steps=0)
    assert result.strategy.score(chsh()) == Fraction(1, 2)
    assert original.score(chsh()) > .85
    assert result.diagnostics['measurements'][1][0]['mixing_weight'] == 1
    assert not result.diagnostics['closeness_or_optimality_claimed']
    with pytest.raises(ExactBudgetExceeded):
        rationalize_strategy(original, budget=ExactBudget(max_psd_size=7))


def test_trace_zero_after_rounding_has_exact_normalized_repair():
    import numpy as np
    from nonlocalgames.quantum import QuantumStrategy
    from nonlocalgames.quantum.exact.rounding import rationalize_strategy
    numerical = QuantumStrategy(np.eye(3)/3, (((np.eye(3),),),))
    result = rationalize_strategy(numerical, max_denominator=1, repair_steps=0)
    assert result.diagnostics['state']['trace_before_correction'] == 0
    assert result.diagnostics['state']['trace_identity_shift'] == Fraction(1, 3)
    assert result.strategy.state == scale(identity(3), Fraction(1, 3))
    assert result.strategy.probabilities() == (Fraction(1),)


def test_exact_reference_scores_match_independent_julia_constants():
    from nonlocalgames.quantum.exact import exact_chsh_strategy, exact_ghz_strategy
    chsh_reference = exact_chsh_strategy()
    assert chsh_reference.score(chsh()) == Fraction(3880899, 4546756)
    assert exact_ghz_strategy().probabilities() == _ghz_exact().probabilities()
    for party in chsh_reference.measurements:
        for povm in party:
            for effect in povm:
                assert multiply(effect, effect) == effect


def test_claimed_lower_bound_cannot_expand_compact_exponents():
    from nonlocalgames.certificates.quantum import certify_quantum_strategy, verify_quantum_lower_certificate
    from nonlocalgames.quantum.exact import exact_chsh_strategy
    certificate = certify_quantum_strategy(chsh(), exact_chsh_strategy()).witness
    for text in ('1e1000000000', '0.5000', '+1', '2/2'):
        with pytest.raises(ValueError):
            verify_quantum_lower_certificate(chsh(), {**certificate, 'lower_bound': text})


def test_identity_entry_guard_precedes_matrix_materialization(monkeypatch):
    import nonlocalgames.quantum.exact.algebra as algebra
    monkeypatch.setattr(algebra, 'RationalMatrix', lambda *a, **kw: pytest.fail('must guard before constructing'))
    with pytest.raises(ExactBudgetExceeded, match='max_matrix_entries'):
        algebra.identity(100, budget=ExactBudget(max_joint_dimension=100, max_matrix_entries=3))
