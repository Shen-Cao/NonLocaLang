"""Exact-quantum scalar policy over the shared bounded rational parser."""
from fractions import Fraction
from numbers import Integral
from operator import index

from ...core.scalar_wire import ScalarBudgetExceeded, parse_rational, scalar_bits, scalar_string
from .budgets import ExactBudgetExceeded

__all__ = ['exact_scalar', 'scalar_bits', 'scalar_string']


def exact_scalar(value, *, max_bits=4096):
    """Accept explicit exact numbers and canonical fraction strings, never floats.

    Native Fraction inputs are copied after bounding their integer components.
    The shared wire parser checks canonical text and other exact input types.
    Its character budget grows with the configured bit limit.
    """
    if type(value) is Fraction and type(max_bits) is int and max_bits > 0:
        numerator, denominator = value.numerator, value.denominator
        if (type(numerator) is int and type(denominator) is int and denominator > 0
                and numerator.bit_length() <= max_bits and denominator.bit_length() <= max_bits):
            return Fraction(numerator, denominator)
    if isinstance(max_bits, bool) or not isinstance(max_bits, Integral) or max_bits < 1:
        raise ValueError('max_bits must be a positive integer')
    if isinstance(value, bool) or not isinstance(value, (Fraction, Integral, str)):
        raise TypeError('Use integers, Fractions, or canonical rational strings for exact scalars')
    max_bits = index(max_bits)
    digits = (max_bits*30103)//100000 + 1
    try:
        return parse_rational(value, max_bits=max_bits, max_chars=2*digits+2, canonical=True)
    except ScalarBudgetExceeded as error:
        raise ExactBudgetExceeded('Exact scalar exceeds max_scalar_bits') from error
