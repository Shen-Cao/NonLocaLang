"""Adversarial exact checks of XOR SDP dual scaling and PSD verification."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from nonlocalgames import AnalysisTask, Game, ResourceSpec, Scenario
from nonlocalgames.games import chsh
from nonlocalgames.certificates.linalg import verify_psd
from nonlocalgames.certificates.xor import verify_xor_certificate, verify_xor_dual
from nonlocalgames.certificates.xor_discovery import certify_xor_upper_bound


def test_exact_psd_zero_pivots_and_rank_deficiency():
    assert verify_psd([[0, 0], [0, 0]]).rank == 0
    assert verify_psd([[0, 0, 0], [0, 2, 1], [0, 1, Fraction(1, 2)]]).rank == 1
    assert verify_psd([[1, 1, 0], [1, 2, 1], [0, 1, 1]]).rank == 2
    for matrix in (
        [[0, 1], [1, 0]],
        [[1, 1, 0], [1, 1, 1], [0, 1, 2]],
        [[1, 1], [1, 1-Fraction(1, 10**100)]],
        [[0, Fraction(1, 10**300)], [Fraction(1, 10**300), 1]],
        [[1, 0], [1, 1]],
    ):
        with pytest.raises(ValueError):
            verify_psd(matrix)
    with pytest.raises(TypeError):
        verify_psd([[1.0, 0], [0, 1]])


def test_single_question_signed_xor_has_exact_dual_scaling():
    game = Game.from_rule(Scenario((1, 1), (2, 2)),
                          lambda x, a: 1 if a[0] == a[1] else -1)
    result = verify_xor_dual(game, (Fraction(1, 2), Fraction(1, 2)))
    assert result.value == 1
    assert result.guarantee == 'certified_upper_bound'
    with pytest.raises(ValueError):
        verify_xor_dual(game, (Fraction(1, 4), Fraction(1, 4)))
    negative = Game.from_rule(game.scenario, lambda x, a: Fraction(-7, 3))
    result = verify_xor_dual(negative, (0, 0))
    assert result.value == Fraction(-7, 3)
    assert verify_xor_certificate(negative, result.witness).value == result.value


def _asymmetric_game():
    scenario = Scenario((2, 3), (2, 2))
    def rule(x, a):
        return Fraction(2*x[0] + x[1] - 3, 4) if a[0] == a[1] else Fraction(x[0]-2*x[1]+1, 5)
    return Game.from_rule(scenario, rule, distribution={(0, 0): Fraction(1, 5),
        (0, 2): Fraction(2, 5), (1, 1): Fraction(2, 5)}, name='asymmetric signed promise')


def test_signed_promised_asymmetric_questions_against_known_feasible_dual():
    game = _asymmetric_game()
    diagonal = [Fraction(0)]*5
    constant = Fraction(0)
    for x in game.inputs:
        even, odd = game.reward(x, (0, 0)), game.reward(x, (0, 1))
        entry = game.probability(x)*(even-odd)/4
        diagonal[x[0]] += abs(entry)
        diagonal[2+x[1]] += abs(entry)
        constant += game.probability(x)*(even+odd)/2
    result = verify_xor_dual(game, diagonal)
    assert result.value == constant + sum(diagonal)
    assert result.value >= AnalysisTask(game, ResourceSpec('local')).solve().value
    assert verify_xor_certificate(replace(game, name='same verifier renamed'), result.witness).value == result.value


def test_saved_certificate_rejects_changed_bound_diagonal_and_game():
    game = chsh()
    result = verify_xor_dual(game, [Fraction(9, 100)]*4)
    assert result.value == Fraction(43, 50)
    for edit in (
        lambda data: data.update(upper_bound='3/4'),
        lambda data: data.update(diagonal=['1/16']*4),
        lambda data: data.update(game_fingerprint='0'*64),
        lambda data: data.update(extra_field=True),
    ):
        changed = deepcopy(result.witness)
        edit(changed)
        with pytest.raises(ValueError):
            verify_xor_certificate(game, changed)
    different = Game(game.scenario, game.distribution, tuple(1-v for v in game.payoff))
    with pytest.raises(ValueError):
        verify_xor_certificate(different, result.witness)


def test_verifier_uses_no_solver_or_floating_eigenvalue_check(monkeypatch):
    import cvxpy as cp
    import numpy as np
    def forbidden(*args, **kwargs):
        pytest.fail('exact certificate checking must not call numerical algebra or optimization')
    monkeypatch.setattr(cp.Problem, 'solve', forbidden)
    monkeypatch.setattr(np.linalg, 'eigvalsh', forbidden)
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    monkeypatch.setattr(np.linalg, 'cholesky', forbidden)
    result = verify_xor_dual(chsh(), [Fraction(9, 100)]*4)
    assert verify_xor_certificate(chsh(), result.witness).value == Fraction(43, 50)


def test_discovered_chsh_bound_is_rationally_above_the_algebraic_optimum():
    result = certify_xor_upper_bound(chsh(), solver='SCS', eps=1e-8)
    assert isinstance(result.value, Fraction)
    assert result.value >= Fraction(1, 2)
    assert (result.value-Fraction(1, 2))**2 >= Fraction(1, 8)
    assert result.value < Fraction(854, 1000)
    assert result.guarantee == 'certified_upper_bound'
    assert verify_xor_certificate(chsh(), result.witness).value == result.value


def test_resource_dispatch_does_not_claim_dimension_constrained_characterization():
    for kind in ('quantum_finite', 'quantum_closure', 'quantum_commuting'):
        assert 'xor_certified' in AnalysisTask(chsh(), ResourceSpec(kind)).capabilities()
    from nonlocalgames.networks import SourceNetwork
    restrictions = (
        ResourceSpec('quantum_finite', (2, 2)),
        ResourceSpec('quantum_finite', (2, 2), state_constraint='ppt', ppt_cuts=((0,),)),
        ResourceSpec('quantum_finite', network=SourceNetwork(2, ((0, 1),)), shared_randomness='forbidden'),
    )
    for resource in restrictions:
        task = AnalysisTask(chsh(), resource)
        with pytest.raises(NotImplementedError):
            task.solve(method='xor_certified')


def test_bad_numerical_dual_is_repaired_before_any_certificate_claim(monkeypatch):
    from nonlocalgames.analysis import AnalysisResult
    import nonlocalgames.analysis.xor as numerical
    bogus = AnalysisResult(-1000.0, 'quantum', 'bogus', 'numerical_optimum', 'optimal', {'solver': 'audit_fake'})
    monkeypatch.setattr(numerical, '_solve_xor', lambda *args, **kwargs: (bogus, [-100.0]*4))
    result = certify_xor_upper_bound(chsh(), repair_steps=35)
    assert result.value >= Fraction(1, 2)
    assert (result.value-Fraction(1, 2))**2 >= Fraction(1, 8)
    assert result.value < Fraction(854, 1000)
    assert result.diagnostics['diagonal_shift'] > 100
    assert result.diagnostics['repair_bound_increase'] == 4*result.diagnostics['diagonal_shift']
    assert verify_xor_certificate(chsh(), result.witness).value == result.value


def test_pure_verifier_retains_rationals_that_underflow_numerical_discovery():
    tiny = Fraction(1, 10**400)
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: tiny if a[0] == a[1] else 0)
    result = verify_xor_dual(game, (tiny/4, tiny/4))
    assert result.value == tiny
    with pytest.raises(ValueError, match='underflow'):
        certify_xor_upper_bound(game, solver='SCS')
