"""Bounded exact scalar decoding for data boundaries, independent of quantum modules."""
from fractions import Fraction
from numbers import Integral
from operator import index
import re


class ScalarBudgetExceeded(ValueError):
    """A scalar's declared text or integer-size budget was exceeded."""


_DIGITS = r'[0-9](?:_?[0-9])*'
_WS = r'[ \t\n\r\f\v]*'
_FRACTION = re.compile(rf'(?P<sign>[+-]?)(?P<num>{_DIGITS}){_WS}/{_WS}(?P<den>{_DIGITS})\Z', re.ASCII)
_DECIMAL = re.compile(
    rf'(?P<sign>[+-]?)(?:(?P<whole>{_DIGITS})(?:\.(?P<frac>{_DIGITS})?)?|\.(?P<onlyfrac>{_DIGITS}))'
    rf'(?:[eE](?P<esign>[+-]?)(?P<exp>{_DIGITS}))?\Z', re.ASCII)
_CANONICAL = re.compile(r'-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?\Z', re.ASCII)
_ASCII_WHITESPACE = ' \t\n\r\f\v'


def _limit(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return index(value)


def _parts(value):
    if isinstance(value, bool):
        raise TypeError('Exact scalar data do not accept booleans')
    if isinstance(value, Fraction):
        numerator, denominator = index(value.numerator), index(value.denominator)
    elif isinstance(value, Integral):
        numerator, denominator = index(value), 1
    else:
        raise TypeError('Use integers, Fractions, or exact scalar strings; floating-point and implicit conversions are forbidden')
    if denominator == 0:
        raise ValueError('A rational denominator must be nonzero')
    return numerator, denominator


def scalar_bits(value):
    """Measure exact integer components without coercing through floating point."""
    numerator, denominator = _parts(value)
    return max(abs(numerator).bit_length(), abs(denominator).bit_length())


def _integer_string(number):
    if number == 0:
        return '0'
    negative, number = number < 0, abs(number)
    chunks = []
    while number:
        number, remainder = divmod(number, 1_000_000_000)
        chunks.append(remainder)
    return ('-' if negative else '') + str(chunks[-1]) + ''.join(f'{c:09d}' for c in reversed(chunks[:-1]))


def scalar_string(value):
    """Emit reduced n[/d] text without Python's global decimal-digit limit.

    The caller must bound values before exposing this formatter to untrusted data.
    Formatting does not apply an implicit precision or size limit.
    """
    numerator, denominator = _parts(value)
    normalized = Fraction(numerator, denominator)
    text = _integer_string(normalized.numerator)
    return text if normalized.denominator == 1 else text + '/' + _integer_string(normalized.denominator)


def _digit_limit(max_bits):
    # 30103/100000 is an upper bound for log10(2); exact bit checks follow conversion.
    return (max_bits*30103)//100000 + 1


def _decimal_integer(digits):
    number = 0
    for start in range(0, len(digits), 9):
        chunk = digits[start:start+9]
        number = number*10**len(chunk) + int(chunk)
    return number


def _bounded_integer(digits, max_bits):
    digits = digits.lstrip('0') or '0'
    if len(digits) > _digit_limit(max_bits):
        raise ScalarBudgetExceeded('Exact scalar component exceeds max_bits')
    number = _decimal_integer(digits)
    if number.bit_length() > max_bits:
        raise ScalarBudgetExceeded('Exact scalar component exceeds max_bits')
    return number


def _exponent(digits, sign, limit):
    if digits is None:
        return 0
    digits = digits.replace('_', '').lstrip('0') or '0'
    bound = _integer_string(limit)
    if len(digits) > len(bound) or (len(digits) == len(bound) and digits > bound):
        raise ScalarBudgetExceeded('Decimal exponent exceeds bounded expansion for max_bits')
    value = _decimal_integer(digits)
    return -value if sign == '-' else value


def _power10(exponent, max_bits):
    if exponent + 1 > _digit_limit(max_bits):
        raise ScalarBudgetExceeded('Decimal power exceeds max_bits')
    value = 10**exponent
    if value.bit_length() > max_bits:
        raise ScalarBudgetExceeded('Decimal power exceeds max_bits')
    return value


def _legacy_string(text, max_bits):
    text = text.strip(_ASCII_WHITESPACE)
    if '/' in text:
        match = _FRACTION.fullmatch(text)
        if match is None:
            raise ValueError('Expected an ASCII integer fraction or finite decimal string')
        numerator = _bounded_integer(match['num'].replace('_', ''), max_bits)
        denominator = _bounded_integer(match['den'].replace('_', ''), max_bits)
        if denominator == 0:
            raise ValueError('A rational denominator must be nonzero')
        return Fraction(-numerator if match['sign'] == '-' else numerator, denominator)
    match = _DECIMAL.fullmatch(text)
    if match is None:
        raise ValueError('Expected an ASCII integer fraction or finite decimal string')
    whole = (match['whole'] or '').replace('_', '')
    fraction = (match['frac'] or match['onlyfrac'] or '').replace('_', '')
    digits = whole + fraction
    exponent = _exponent(match['exp'], match['esign'], _digit_limit(max_bits) + len(digits))
    significant = digits.lstrip('0')
    if not significant:
        return Fraction(0)
    shift = exponent - len(fraction)
    trailing = len(significant) - len(significant.rstrip('0'))
    significant, shift = significant.rstrip('0'), shift + trailing
    if shift >= 0 and len(significant) + shift > _digit_limit(max_bits):
        raise ScalarBudgetExceeded('Expanded decimal numerator exceeds max_bits')
    numerator = _bounded_integer(significant, max_bits)
    denominator = 1
    if shift >= 0:
        numerator *= _power10(shift, max_bits)
        if numerator.bit_length() > max_bits:
            raise ScalarBudgetExceeded('Expanded decimal numerator exceeds max_bits')
    else:
        denominator = _power10(-shift, max_bits)
    return Fraction(-numerator if match['sign'] == '-' else numerator, denominator)


def parse_rational(value, *, max_bits=4096, max_chars=8192, canonical=False):
    """Decode exact data without unbounded exponent expansion or implicit rounding.

    Legacy text uses ASCII digits, signs, outer ASCII whitespace, underscores
    between digits, slash whitespace, and finite decimal/exponent notation.
    Canonical text must be a reduced integer/fraction with no extra spelling.
    Supplied integer components and decimal powers are conservatively bounded
    before Fraction construction: a reducible noncanonical spelling may exceed
    the budget even when its reduced value fits. Decimal trailing zeros may be
    cancelled textually first. Excessive exponents are rejected even for zero.
    These are data-size limits, not a bound on all later exact arithmetic work.
    """
    max_bits, max_chars = _limit(max_bits, 'max_bits'), _limit(max_chars, 'max_chars')
    if not isinstance(canonical, bool):
        raise TypeError('canonical must be a Boolean')
    if isinstance(value, str):
        if len(value) > max_chars:
            raise ScalarBudgetExceeded('Exact scalar text exceeds max_chars')
        if not value.isascii():
            raise ValueError('Exact scalar wire text must use ASCII characters')
        if canonical:
            if _CANONICAL.fullmatch(value) is None:
                raise ValueError('Expected a reduced canonical rational string')
            parts = value.split('/')
            numerator = _bounded_integer(parts[0].lstrip('-'), max_bits)
            denominator = _bounded_integer(parts[1], max_bits) if len(parts) == 2 else 1
            result = Fraction(-numerator if value.startswith('-') else numerator, denominator)
            if scalar_string(result) != value:
                raise ValueError('Expected a reduced canonical rational string')
        else:
            result = _legacy_string(value, max_bits)
    else:
        numerator, denominator = _parts(value)
        if max(abs(numerator).bit_length(), abs(denominator).bit_length()) > max_bits:
            raise ScalarBudgetExceeded('Exact scalar components exceed max_bits')
        result = Fraction(numerator, denominator)
    if scalar_bits(result) > max_bits:
        raise ScalarBudgetExceeded('Exact scalar exceeds max_bits')
    return result
