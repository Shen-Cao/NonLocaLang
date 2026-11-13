"""Bounded portable wire scalars preserve exact values without expansion surprises."""
from fractions import Fraction

import numpy as np
import pytest

from nonlocalgames.core.scalar_wire import (ScalarBudgetExceeded, parse_rational,
                                             scalar_bits, scalar_string)


@pytest.mark.parametrize('text, expected', [
    ('0', Fraction(0)), ('-0', Fraction(0)), ('+000', Fraction(0)),
    ('  +6 / 8\t', Fraction(3, 4)), ('1_000/2_000', Fraction(1, 2)),
    ('-1_0 / 0_2', Fraction(-5)), ('1.25', Fraction(5, 4)),
    ('-.125', Fraction(-1, 8)), ('.5', Fraction(1, 2)), ('2.', Fraction(2)),
    ('7e-6', Fraction(7, 1_000_000)), ('1.2_5E+2', Fraction(125)),
    ('1_2.3_4e+0_2', Fraction(1234)), ('000.00100', Fraction(1, 1000)),
    ('1.200e-2', Fraction(3, 250)), ('10e-1', Fraction(1)),
    ('0.0000', Fraction(0)), ('-00.00e+3', Fraction(0)),
])
def test_portable_legacy_fraction_decimal_and_exponent_forms(text, expected):
    actual = parse_rational(text)
    assert actual == expected
    assert type(actual.numerator) is int and type(actual.denominator) is int


@pytest.mark.parametrize('text', ['-0', '01', '+1', '1/1', '2/4', '0/2',
                                  '1/-2', ' 1', '1.0', '1e0', '1_0'])
def test_canonical_mode_requires_reduced_wire_spelling(text):
    with pytest.raises(ValueError, match='canonical'):
        parse_rational(text, canonical=True)


@pytest.mark.parametrize('value', [Fraction(7, 12), -17, 0, np.int64(2**62),
                                   Fraction(np.int64(2**62), np.int64(3))])
def test_exact_python_inputs_are_normalized_to_builtin_integer_components(value):
    actual = parse_rational(value)
    assert actual == value
    assert type(actual.numerator) is int and type(actual.denominator) is int
    if actual == Fraction(2**62, 3):
        assert actual*actual == Fraction(2**124, 9)


@pytest.mark.parametrize('value', [True, np.bool_(False), .5, np.float64(.5), complex(.5), None, object()])
def test_inexact_or_implicit_scalar_conversions_are_rejected(value):
    with pytest.raises(TypeError):
        parse_rational(value)


@pytest.mark.parametrize('text', ['', ' ', 'nan', 'inf', '-Infinity', '1/0', '0 / 0',
                                  '1//2', '1__2', '1_.2', '1._2', '1e+', '1e_2',
                                  '1e +2', '1 / -2', '1.2/3', '１２', '١/٢', '\u00a01'])
def test_invalid_or_nonascii_wire_text_has_a_normalized_value_error(text):
    with pytest.raises(ValueError) as failure:
        parse_rational(text)
    assert not isinstance(failure.value, ZeroDivisionError)


def test_compact_huge_exponents_reject_before_integer_conversion_or_power(monkeypatch):
    import nonlocalgames.core.scalar_wire as wire

    def forbidden(*args, **kwargs):
        pytest.fail('The exponent budget must fail before integer conversion or powers')

    monkeypatch.setattr(wire, '_decimal_integer', forbidden)
    monkeypatch.setattr(wire, '_power10', forbidden)
    # These are passed only to the guarded new parser, never to Fraction(string).
    for text in ('1e1000000000', '1e-1000000000', '0e1000000000', '1e' + '9'*2000):
        with pytest.raises(ScalarBudgetExceeded, match='exponent'):
            wire.parse_rational(text)


def test_text_budget_precedes_grammar_and_conversion(monkeypatch):
    import nonlocalgames.core.scalar_wire as wire
    monkeypatch.setattr(wire, '_legacy_string', lambda *a: pytest.fail('text size must fail first'))
    with pytest.raises(ScalarBudgetExceeded, match='max_chars'):
        wire.parse_rational('0'*101, max_chars=100)


def test_expanded_powers_and_integer_components_have_explicit_bit_limits():
    with pytest.raises(ScalarBudgetExceeded, match='max_bits'):
        parse_rational('1e1234')
    with pytest.raises(ScalarBudgetExceeded, match='max_bits'):
        parse_rational('1e-1234')
    with pytest.raises(ScalarBudgetExceeded, match='max_bits'):
        parse_rational('1000/2000', max_bits=4)
    with pytest.raises(ScalarBudgetExceeded, match='max_bits'):
        parse_rational('0.125', max_bits=4)
    # Decimal trailing zeros can be cancelled without constructing oversized parts.
    assert parse_rational('1000e-3', max_bits=4) == 1
    assert parse_rational('1e' + '0'*2000 + '1') == 10


def test_exact_bit_boundary_and_large_canonical_roundtrip_without_global_digit_settings():
    limit = 64
    largest = (1 << limit)-1
    assert parse_rational(scalar_string(Fraction(largest)), max_bits=limit, canonical=True) == largest
    with pytest.raises(ScalarBudgetExceeded):
        parse_rational(scalar_string(Fraction(1 << limit)), max_bits=limit, canonical=True)
    with pytest.raises(ScalarBudgetExceeded):
        parse_rational(Fraction(1, 1 << limit), max_bits=limit)
    value = Fraction((1 << 20_000)-1, (1 << 19_999)+1)
    encoded = scalar_string(value)
    assert len(encoded) > 8192
    assert parse_rational(encoded, max_bits=20_000, max_chars=len(encoded), canonical=True) == value
    assert scalar_bits(value) <= 20_000
    with pytest.raises(ScalarBudgetExceeded, match='max_chars'):
        parse_rational(encoded, max_bits=20_000, canonical=True)


@pytest.mark.parametrize('options', [
    {'max_bits': 0}, {'max_bits': True}, {'max_bits': 1.5},
    {'max_chars': 0}, {'max_chars': False}, {'max_chars': 3.5},
])
def test_scalar_budgets_are_positive_integers(options):
    with pytest.raises(ValueError, match='positive integer'):
        parse_rational('1', **options)


def test_canonical_policy_is_an_explicit_boolean_and_formatting_is_exact():
    with pytest.raises(TypeError, match='Boolean'):
        parse_rational('1', canonical='yes')
    for value in (Fraction(-8, 4), Fraction(0), Fraction(2, 3)):
        text = scalar_string(value)
        assert parse_rational(text, canonical=True) == value
    with pytest.raises(TypeError):
        scalar_string(.5)
