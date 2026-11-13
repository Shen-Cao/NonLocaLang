"""Exact-quantum scalar policy over the shared bounded rational parser."""
from fractions import Fraction
from numbers import Integral
from operator import index

from ...core.scalar_wire import ScalarBudgetExceeded, parse_rational, scalar_bits, scalar_string
from .budgets import ExactBudgetExceeded

__all__ = ['exact_scalar', 'scalar_bits', 'scalar_string']


def exact_scalar(value, *, max_bits=4096):
    """Accept explicit exact numbers and canonical fraction strings, never floats.

    The shared core parser handles arithmetic and preconversion bounds. This
    policy retains the exact quantum wire's stricter canonical syntax and maps
    scalar-budget failures to the exact quantum budget exception. The character
    budget grows with an explicitly increased bit limit; it is not fixed at the
    more permissive legacy parser's default character limit.
    """
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
