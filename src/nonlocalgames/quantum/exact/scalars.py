"""Bounded exact scalar parsing; compact exponent notation is never expanded."""
from fractions import Fraction
from numbers import Integral
from operator import index
import re

from .budgets import ExactBudgetExceeded

_CANONICAL = re.compile(r'-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?\Z', re.ASCII)


def scalar_bits(value):
    return max(abs(index(value.numerator)).bit_length(), index(value.denominator).bit_length())


def _decimal_integer(text):
    negative = text.startswith('-')
    digits = text[1:] if negative else text
    number = 0
    for start in range(0, len(digits), 9):
        chunk = digits[start:start+9]
        number = number * 10**len(chunk) + int(chunk)
    return -number if negative else number


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
    """Format a checked Fraction canonically without Python's decimal-digit limit."""
    numerator = _integer_string(index(value.numerator))
    return numerator if value.denominator == 1 else numerator + '/' + _integer_string(index(value.denominator))


def exact_scalar(value, *, max_bits=4096):
    """Accept integers, Fractions, and canonical rational strings, never floats.

    Length is bounded before decimal conversion. Rational strings must already
    be reduced with positive denominator and no whitespace, decimal point,
    exponent, leading plus, leading zeros, or redundant denominator one.
    """
    if isinstance(max_bits, bool) or not isinstance(max_bits, Integral) or max_bits < 1:
        raise ValueError('max_bits must be a positive integer')
    if isinstance(value, bool):
        raise TypeError('Exact scalars do not accept booleans')
    if isinstance(value, Fraction):
        numerator, denominator = index(value.numerator), index(value.denominator)
        if max(abs(numerator).bit_length(), denominator.bit_length()) > max_bits:
            raise ExactBudgetExceeded('Exact scalar exceeds max_scalar_bits')
        result = Fraction(numerator, denominator)
    elif isinstance(value, Integral):
        number = index(value)
        if abs(number).bit_length() > max_bits:
            raise ExactBudgetExceeded('Exact scalar exceeds max_scalar_bits')
        result = Fraction(number)
    elif isinstance(value, str):
        digits = (index(max_bits)*30103)//100000 + 1
        if len(value) > 2*digits + 2:
            raise ExactBudgetExceeded('Exact scalar text exceeds max_scalar_bits')
        if _CANONICAL.fullmatch(value) is None:
            raise ValueError('Expected a canonical rational string')
        parts = value.split('/')
        if any(len(part.lstrip('-')) > digits for part in parts):
            raise ExactBudgetExceeded('Exact scalar text exceeds max_scalar_bits')
        numerator = _decimal_integer(parts[0])
        denominator = _decimal_integer(parts[1]) if len(parts) == 2 else 1
        if max(abs(numerator).bit_length(), denominator.bit_length()) > max_bits:
            raise ExactBudgetExceeded('Exact scalar exceeds max_scalar_bits')
        result = Fraction(numerator, denominator)
        if scalar_string(result) != value:
            raise ValueError('Expected a reduced canonical rational string')
    else:
        raise TypeError('Use integers, Fractions, or canonical rational strings for exact scalars')
    if scalar_bits(result) > max_bits:
        raise ExactBudgetExceeded('Exact scalar exceeds max_scalar_bits')
    return result
