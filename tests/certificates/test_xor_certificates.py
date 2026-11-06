"""Rational XOR upper bounds survive independent checks, tampering, and signed rewards."""
from dataclasses import replace
from fractions import Fraction
import json
import math

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import (certify_xor_upper_bound, classical_value,
                                    verify_xor_certificate, verify_xor_dual, xor_quantum_value)
from nonlocalgames.certificates.xor_discovery import repair_diagonal
from nonlocalgames.certificates.xor_model import cost_matrix, xor_coefficients
from nonlocalgames.games import chsh


def test_chsh_discovery_returns_verified_rational_upper_bound():
    result = certify_xor_upper_bound(chsh(), solver='SCS', eps=1e-8)
    assert result.guarantee == 'certified_upper_bound' and result.diagnostics['certified']
    assert isinstance(result.value, Fraction)
    assert result.value >= Fraction(3, 4)
    assert float(result.value) == pytest.approx((2 + math.sqrt(2))/4, abs=2e-7)
    assert 'lower_bound' not in result.diagnostics
    assert verify_xor_certificate(chsh(), json.loads(json.dumps(result.witness))).value == result.value
    numeric = xor_quantum_value(chsh(), solver='SCS', eps=1e-8)
    assert numeric.guarantee == 'numerical_optimum' and numeric.witness.shape == (4, 4)


def test_pure_verifier_uses_exact_psd_and_is_solver_independent(monkeypatch):
    import nonlocalgames.analysis.xor as numeric
    monkeypatch.setattr(numeric, '_solve_xor', lambda *a, **kw: pytest.fail('must not solve'))
    result = verify_xor_dual(chsh(), ['1/8']*4)
    assert result.value == 1
    assert result.guarantee == 'certified_upper_bound'
    assert verify_xor_certificate(chsh(), result.witness).value == 1
    with pytest.raises(ValueError, match='not PSD'):
        verify_xor_dual(chsh(), [Fraction(1, 16)]*4)
    with pytest.raises(TypeError):
        verify_xor_dual(chsh(), [.125]*4)


def test_certificate_context_bound_and_schema_are_rechecked():
    result = verify_xor_dual(chsh(), [Fraction(1, 8)]*4)
    assert verify_xor_certificate(replace(chsh(), name='renamed'), result.witness).value == 1
    for field, value in [('upper_bound', '3/4'), ('schema', 'future'), ('game_fingerprint', 'wrong')]:
        payload = {**result.witness, field: value}
        with pytest.raises(ValueError):
            verify_xor_certificate(chsh(), payload)
    payload = {**result.witness, 'diagonal': ['1/16']*4}
    with pytest.raises(ValueError, match='not PSD'):
        verify_xor_certificate(chsh(), payload)
    payload = {**result.witness, 'diagonal': [.125]*4}
    with pytest.raises(ValueError, match='rational strings'):
        verify_xor_certificate(chsh(), payload)


def test_signed_rectangular_xor_with_inactive_questions():
    scenario = Scenario((2, 3), (2, 2))
    game = Game.from_rule(scenario, lambda x, a: 3*(a[0] == a[1]) - 2*x[1],
                         distribution={(0, 0): Fraction(1, 3), (1, 2): Fraction(2, 3)})
    constant, weights = xor_coefficients(game)
    matrix = cost_matrix(weights)
    diagonal = tuple(sum(abs(value) for value in row) for row in matrix)
    result = verify_xor_dual(game, diagonal)
    assert result.value == classical_value(game).value == Fraction(1, 3)
    assert result.diagnostics['psd_rank'] < 5
    assert constant == Fraction(-7, 6)


def test_zero_parity_weight_retains_signed_constant_without_fake_lower_bound():
    game = Game.from_rule(Scenario((1, 2), (2, 2)), lambda x, a: -3)
    result = verify_xor_dual(game, [0]*3)
    assert result.value == -3 and result.diagnostics['psd_rank'] == 0
    assert result.guarantee == 'certified_upper_bound'


def test_exact_repair_is_bounded_and_records_its_bound_cost():
    _, weights = xor_coefficients(chsh())
    matrix = cost_matrix(weights)
    diagonal = (Fraction(1, 16),)*4
    repaired, shift, endpoint, steps = repair_diagonal(matrix, diagonal, 20)
    assert 0 < shift <= endpoint and steps == 20
    result = verify_xor_dual(chsh(), repaired)
    assert float(result.value) == pytest.approx((2 + math.sqrt(2))/4, abs=1e-6)
    coarse, shift, endpoint, steps = repair_diagonal(matrix, diagonal, 0)
    assert shift == endpoint and steps == 0
    assert verify_xor_dual(chsh(), coarse).value == 1


def test_discovery_invalid_inputs_and_float_range_fail_explicitly():
    for options in ({'max_denominator': 0}, {'repair_steps': -1}, {'repair_steps': True},
                    {'max_matrix_size': 3}):
        with pytest.raises(ValueError):
            certify_xor_upper_bound(chsh(), **options)
    for scale in (Fraction(10**1000), Fraction(1, 10**1000)):
        game = Game(chsh().scenario, chsh().distribution,
                    tuple(scale * v for v in chsh().payoff))
        with pytest.raises(ValueError, match='float64'):
            xor_quantum_value(game, solver='SCS')
    invalid = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: a[0])
    with pytest.raises(ValueError, match='XOR'):
        verify_xor_dual(invalid, [1, 1])


def test_dual_length_is_bounded_before_converting_an_iterable():
    consumed = []

    def oversized():
        for i in range(5):
            consumed.append(i)
            yield object()
        pytest.fail("must not consume beyond the expected diagonal size plus one")

    with pytest.raises(ValueError, match="length"):
        verify_xor_dual(chsh(), oversized())
    assert consumed == list(range(5))
    payload = verify_xor_dual(chsh(), [Fraction(1, 8)]*4).witness
    payload['diagonal'] = [object()]*5
    with pytest.raises(ValueError, match="length"):
        verify_xor_certificate(chsh(), payload)


def test_off_promise_nonparity_rewards_do_not_change_the_xor_objective():
    scenario = Scenario((2, 2), (2, 2))
    promised = {(0, 0): Fraction(1, 2), (1, 1): Fraction(1, 2)}
    parity = Game.from_rule(scenario, lambda x, a: a[0] == a[1], distribution=promised)
    irregular = Game.from_rule(scenario, lambda x, a: (a[0] == a[1]) if x[0] == x[1]
                               else 1000*a[0] - 7*a[1], distribution=promised)
    clean = certify_xor_upper_bound(parity, solver='SCS', eps=1e-8)
    checked = verify_xor_dual(irregular, clean.witness['diagonal'])
    assert checked.value == clean.value
    assert checked.diagnostics['matrix_size'] == 4
    assert checked.witness['game_fingerprint'] != clean.witness['game_fingerprint']
    with pytest.raises(ValueError, match='different verifier'):
        verify_xor_certificate(irregular, clean.witness)
    numeric = xor_quantum_value(irregular, solver='SCS', eps=1e-8)
    assert numeric.value == pytest.approx(1, abs=1e-7)
