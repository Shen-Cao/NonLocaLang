"""XOR scalar-wire limits precede decoding, compilation, and numerical discovery."""
from fractions import Fraction

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.analysis import AnalysisResult
from nonlocalgames.certificates.xor import verify_xor_certificate, verify_xor_dual
from nonlocalgames.certificates.xor_discovery import certify_xor_upper_bound, repair_diagonal
from nonlocalgames.certificates.xor_model import cost_matrix, xor_coefficients
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded, scalar_bits
from nonlocalgames.games import chsh


@pytest.mark.parametrize('field', ['diagonal', 'upper_bound'])
def test_compact_exponents_fail_before_hashing_or_compiling(monkeypatch, field):
    import nonlocalgames.certificates.xor as checker
    import nonlocalgames.core.scalar_wire as wire
    payload = verify_xor_dual(chsh(), [Fraction(1, 8)]*4).witness
    payload['upper_bound'] = '1/1'
    if field == 'diagonal':
        payload[field] = ['1e-1000000000', '1/8', '1/8', '1/8']
    else:
        payload[field] = '1e1000000000'

    def forbidden(*args, **kwargs):
        pytest.fail('wire limits must reject before exact compilation, hashing, or powers')

    monkeypatch.setattr(checker, 'xor_coefficients', forbidden)
    monkeypatch.setattr(checker, 'game_fingerprint', forbidden)
    monkeypatch.setattr(wire, '_power10', forbidden)
    with pytest.raises(ScalarBudgetExceeded):
        checker.verify_xor_certificate(chsh(), payload)


def test_game_values_including_unused_rewards_are_checked_before_compilation(monkeypatch):
    import nonlocalgames.certificates.xor as checker
    import nonlocalgames.analysis.xor as numerical
    game = Game.from_rule(chsh().scenario, lambda x, a: (a[0] == a[1]) if x == (0, 0) else 1 << 200,
                          distribution={(0, 0): 1})

    def forbidden(*args, **kwargs):
        pytest.fail('raw game scalar budgets must fail before compilation or solving')

    class Untouched:
        def __iter__(self):
            pytest.fail('raw game scalar budgets must precede witness consumption')

    monkeypatch.setattr(checker, 'xor_coefficients', forbidden)
    monkeypatch.setattr(checker, 'game_fingerprint', forbidden)
    monkeypatch.setattr(numerical, '_solve_xor', forbidden)
    with pytest.raises(ScalarBudgetExceeded):
        checker.verify_xor_dual(game, Untouched(), max_scalar_bits=128)
    with pytest.raises(ScalarBudgetExceeded):
        certify_xor_upper_bound(game, max_scalar_bits=128)


def test_bounded_legacy_spellings_and_character_budget_remain_explicit():
    payload = verify_xor_dual(chsh(), [Fraction(1, 8)]*4).witness
    payload['diagonal'] = [' .125 ', '1.25e-1', '+2 / 16', '0.125']
    payload['upper_bound'] = '1.0'
    assert verify_xor_certificate(chsh(), payload).value == 1
    with pytest.raises(ScalarBudgetExceeded, match='max_chars'):
        verify_xor_certificate(chsh(), payload, max_scalar_chars=3)
    payload['upper_bound'] = '1/0'
    with pytest.raises(ValueError, match='nonzero') as failure:
        verify_xor_certificate(chsh(), payload)
    assert not isinstance(failure.value, ZeroDivisionError)


def test_explicit_large_scalar_budgets_keep_portable_canonical_output():
    value = Fraction((1 << 16_384) + 1)
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: value)
    with pytest.raises(ScalarBudgetExceeded):
        verify_xor_dual(game, [0, 0])
    result = verify_xor_dual(game, [0, 0], max_scalar_bits=20_000, max_scalar_chars=20_000)
    assert result.value == value
    assert len(result.witness['upper_bound']) > 4300
    assert verify_xor_certificate(game, result.witness,
                                   max_scalar_bits=20_000, max_scalar_chars=20_000).value == value


def test_derived_slack_does_not_acquire_a_hidden_default_scalar_cap():
    q, r = (1 << 3000) + 1, (1 << 3001) + 1
    game = Game.from_rule(Scenario((2, 1), (2, 2)),
                          lambda x, a: Fraction(1, r) if a[0] == a[1] else 0,
                          distribution={(0, 0): Fraction(1, q), (1, 0): Fraction(q-1, q)})
    _, weights = xor_coefficients(game)
    assert max(scalar_bits(value) for row in cost_matrix(weights) for value in row) > 4096
    result = verify_xor_dual(game, [1, 1, 1])
    assert result.value == Fraction(3) + Fraction(1, 2*r)
    assert verify_xor_certificate(game, result.witness).value == result.value


def test_newly_serialized_derived_upper_bound_may_require_a_larger_input_budget():
    game = Game.from_rule(Scenario((1, 1), (2, 2)), lambda x, a: 0)
    result = verify_xor_dual(game, [3, 3], max_scalar_bits=2)
    assert result.value == 6
    with pytest.raises(ScalarBudgetExceeded):
        verify_xor_certificate(game, result.witness, max_scalar_bits=2)
    assert verify_xor_certificate(game, result.witness, max_scalar_bits=3).value == 6


def test_discovery_propagates_raw_scalar_budget_failures_without_repair(monkeypatch):
    import nonlocalgames.analysis.xor as numerical
    import nonlocalgames.certificates.xor_discovery as discovery
    result = AnalysisResult(0.0, 'quantum', 'mock', 'numerical_optimum', 'optimal', {'solver': 'mock'})
    monkeypatch.setattr(numerical, '_solve_xor', lambda *a, **k: (result, [1024.0]*4))
    monkeypatch.setattr(discovery, 'repair_diagonal', lambda *a, **k: pytest.fail('must not repair around a budget'))
    with pytest.raises(ScalarBudgetExceeded):
        discovery.certify_xor_upper_bound(chsh(), max_scalar_bits=8)


def test_psd_budget_failures_are_not_treated_as_negative_directions(monkeypatch):
    import nonlocalgames.certificates.xor_discovery as discovery

    def exhausted(*args, **kwargs):
        raise ScalarBudgetExceeded('synthetic arithmetic budget')

    monkeypatch.setattr(discovery, '_verify_psd_prepared', exhausted)
    _, weights = xor_coefficients(chsh())
    with pytest.raises(ScalarBudgetExceeded, match='synthetic'):
        repair_diagonal(cost_matrix(weights), (Fraction(1, 8),)*4, 3)
