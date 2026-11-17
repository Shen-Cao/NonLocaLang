"""Exact quantum policy delegates to the bounded core wire parser without loosening syntax."""
from decimal import Decimal
from fractions import Fraction

import pytest

from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded, parse_rational
from nonlocalgames.quantum.exact.budgets import ExactBudgetExceeded
from nonlocalgames.quantum.exact.scalars import exact_scalar, scalar_bits, scalar_string


@pytest.mark.parametrize('text, expected', [
    ('0', Fraction(0)), ('1', Fraction(1)), ('-1', Fraction(-1)),
    ('1/2', Fraction(1, 2)), ('-17/19', Fraction(-17, 19)),
])
def test_canonical_numbers_have_identical_core_and_exact_meaning(text, expected):
    assert parse_rational(text, canonical=True) == exact_scalar(text) == expected
    assert scalar_string(expected) == text


@pytest.mark.parametrize('text, expected', [
    ('0.125', Fraction(1, 8)), ('.5', Fraction(1, 2)), ('1.', Fraction(1)),
    ('1e3', Fraction(1000)), ('-1.25e-2', Fraction(-1, 80)),
    ('  +001_200 / 0_0_3 ', Fraction(400)), ('-2/4', Fraction(-1, 2)),
    ('1_2.5_0e-1', Fraction(5, 4)), ('1e0003', Fraction(1000)),
])
def test_legacy_decimal_forms_are_bounded_but_not_exact_quantum_wire(text, expected):
    assert parse_rational(text) == expected
    with pytest.raises(ValueError):
        exact_scalar(text)


@pytest.mark.parametrize('text', ['01', '+1', '-0', '1/1', '2/4', '1 /2',
                                  '1_000', '1.0', '1e0'])
def test_exact_canonical_grammar_does_not_gain_legacy_aliases(text):
    with pytest.raises(ValueError):
        exact_scalar(text)


@pytest.mark.parametrize('text', ['١/٢', '１/２', '\u00a01/2', '1/2\u2009', '−1/2',
                                  '_1', '1_', '1__0', '1/_2', '1_e1', '1e_1', '.', '1e'])
def test_unicode_and_malformed_digit_separators_are_not_reinterpreted(text):
    with pytest.raises(ValueError):
        parse_rational(text)
    with pytest.raises(ValueError):
        exact_scalar(text)


def test_explicit_bit_limit_expands_exact_character_budget():
    value = Fraction((1 << 32000)-1, (1 << 31999)-1)
    encoded = scalar_string(value)
    assert len(encoded) > 8192
    with pytest.raises(ScalarBudgetExceeded):
        parse_rational(encoded, max_bits=32000, canonical=True)
    assert exact_scalar(encoded, max_bits=32000) == value
    assert exact_scalar('-'+encoded, max_bits=32000) == -value
    assert scalar_bits(value) == 32000


@pytest.mark.parametrize('bits', [1, 2, 3, 4, 8, 16, 64, 4096])
def test_canonical_bit_boundaries_are_not_rejected_by_decimal_estimates(bits):
    maximum = (1 << bits)-1
    assert exact_scalar(scalar_string(Fraction(maximum)), max_bits=bits) == maximum
    with pytest.raises(ExactBudgetExceeded):
        exact_scalar(1 << bits, max_bits=bits)


def test_budget_errors_are_translated_without_reclassifying_malformed_scalars():
    with pytest.raises(ExactBudgetExceeded) as caught:
        exact_scalar('9'*20, max_bits=8)
    assert isinstance(caught.value.__cause__, ScalarBudgetExceeded)
    with pytest.raises(ValueError) as malformed:
        exact_scalar('1e3', max_bits=8)
    assert not isinstance(malformed.value, ExactBudgetExceeded)
    for value in (True, 0.5, 1+0j, Decimal('0.5')):
        with pytest.raises(TypeError):
            exact_scalar(value)
    for budget in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            exact_scalar('1', max_bits=budget)


def test_numpy_fraction_components_remain_unbounded_python_integers():
    import numpy as np
    supplied = Fraction(np.int64(1), np.int64(2**40))
    value = exact_scalar(supplied)
    assert type(value.numerator) is int and type(value.denominator) is int
    second = exact_scalar(Fraction(np.int64(1), np.int64(2**40-1)))
    assert value*second == Fraction(1, 2**40*(2**40-1))


def test_native_fraction_copy_normalizes_only_after_component_limits_and_preserves_errors():
    import numpy as np
    supplied = Fraction(1, 2)
    # Simulate a noncanonical pair, including invalid data from foreign producers.
    supplied._numerator, supplied._denominator = -6, 8
    normalized = exact_scalar(supplied, max_bits=4)
    assert normalized is not supplied
    assert (normalized.numerator, normalized.denominator) == (-3, 4)
    with pytest.raises(ExactBudgetExceeded) as failure:
        exact_scalar(supplied, max_bits=3)
    assert isinstance(failure.value.__cause__, ScalarBudgetExceeded)
    with pytest.raises(ValueError, match='positive integer'):
        exact_scalar(supplied, max_bits=True)
    supplied._denominator = -8
    assert exact_scalar(supplied, max_bits=np.int64(4)) == Fraction(3, 4)
    supplied._denominator = 0
    with pytest.raises(ValueError, match='nonzero'):
        exact_scalar(supplied)
    assert normalized == Fraction(-3, 4)


def test_native_fraction_copy_preserves_chsh_evidence_identity():
    from nonlocalgames import ResourceSpec
    from nonlocalgames.analysis import certify_quantum_strategy
    from nonlocalgames.claims import make_evidence
    from nonlocalgames.games import chsh
    from nonlocalgames.quantum.exact import exact_chsh_strategy
    game = chsh()
    certificate = certify_quantum_strategy(game, exact_chsh_strategy())
    artifact = make_evidence(game, ResourceSpec('quantum_finite'), 'quantum_strategy', certificate.witness)
    assert certificate.value == Fraction(3880899, 4546756)
    assert artifact.artifact_id == 'nonlocalgames.evidence.v1:0ea54b39168f0b2c50a7d900c0062fc7a0ea622ea03a534446c8de7068c8d60e'


def test_bridge_uses_one_core_parser_with_explicit_canonical_policy(monkeypatch):
    import nonlocalgames.quantum.exact.scalars as bridge
    calls = []
    original = bridge.parse_rational
    def inspect(value, **options):
        calls.append(options)
        return original(value, **options)
    monkeypatch.setattr(bridge, 'parse_rational', inspect)
    assert bridge.exact_scalar('1/2', max_bits=32000) == Fraction(1, 2)
    assert calls == [{'max_bits': 32000, 'max_chars': 2*((32000*30103)//100000+1)+2,
                      'canonical': True}]


@pytest.mark.parametrize('text', ['1e1000000000', '1e-1000000000', '0e1000000000',
                                  '1e'+'9'*1000, '1e'+'0'*1000+'1000000000'])
def test_excessive_exponents_are_rejected_before_integer_conversion_or_powers(monkeypatch, text):
    import nonlocalgames.core.scalar_wire as wire
    def forbidden(*args, **kwargs):
        pytest.fail('excessive exponent must be rejected lexically before expansion')
    monkeypatch.setattr(wire, '_decimal_integer', forbidden)
    monkeypatch.setattr(wire, '_power10', forbidden)
    with pytest.raises(ScalarBudgetExceeded):
        parse_rational(text)
    with pytest.raises(ValueError):
        exact_scalar(text)


def test_component_budgets_precede_reduction_but_decimal_zero_cancellation_is_textual(monkeypatch):
    import nonlocalgames.core.scalar_wire as wire
    original_integer, original_power = wire._decimal_integer, wire._power10
    def small_integer(digits):
        assert len(digits) <= 5
        return original_integer(digits)
    def small_power(exponent, max_bits):
        assert exponent <= 4
        return original_power(exponent, max_bits)
    monkeypatch.setattr(wire, '_decimal_integer', small_integer)
    monkeypatch.setattr(wire, '_power10', small_power)
    with pytest.raises(ScalarBudgetExceeded):
        parse_rational('9'*100 + '/' + '9'*100, max_bits=16)
    assert parse_rational('1'+'0'*300+'e-300', max_bits=16) == 1
    assert parse_rational('-0.000e+3', max_bits=16) == 0


@pytest.mark.parametrize('text', ['1/0', '0/0', '-1/00', '0/000_000', '0/1e1000000000'])
def test_zero_denominators_and_fraction_exponent_mixtures_are_invalid(text):
    with pytest.raises(ValueError):
        parse_rational(text)
    with pytest.raises(ValueError):
        exact_scalar(text)


def test_quantum_realification_forwards_an_explicitly_raised_scalar_budget():
    from nonlocalgames.quantum.exact import ExactBudget, RationalMatrix, verify_psd_complex
    value = Fraction(1, (1 << 16000)-1)
    matrix = RationalMatrix(((value,),), budget=ExactBudget(max_scalar_bits=16000))
    assert verify_psd_complex(matrix).rank == 2
    with pytest.raises(ExactBudgetExceeded):
        verify_psd_complex(matrix, budget=ExactBudget(max_scalar_bits=4096))


def test_quantum_psd_boundary_never_turns_scalar_budget_failure_into_nonpositivity(monkeypatch):
    import nonlocalgames.quantum.exact.matrix as matrix_module
    from nonlocalgames.quantum.exact import RationalMatrix, verify_psd_complex
    matrix = RationalMatrix(((1,),))
    def exhausted(*args, **kwargs):
        raise ScalarBudgetExceeded('injected scalar budget failure')
    monkeypatch.setattr(matrix_module, 'verify_psd', exhausted)
    with pytest.raises(ExactBudgetExceeded) as failure:
        verify_psd_complex(matrix)
    assert isinstance(failure.value.__cause__, ScalarBudgetExceeded)
