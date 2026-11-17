"""Real numbers a+b√2 with rational coefficients and the positive embedding of √2."""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from operator import index

from ..core.scalar_wire import parse_rational, scalar_string


def _rational(value):
    if isinstance(value, bool) or not isinstance(value, (Fraction, Integral)):
        raise TypeError("Quadratic coefficients require exact integers or Fractions")
    if isinstance(value, Fraction):
        return Fraction(index(value.numerator), index(value.denominator))
    return Fraction(index(value))


@dataclass(frozen=True, eq=False)
class Quadratic:
    a: Fraction
    b: Fraction = Fraction(0)

    def __post_init__(self):
        object.__setattr__(self, "a", _rational(self.a))
        object.__setattr__(self, "b", _rational(self.b))

    def __eq__(self, other):
        if isinstance(other, Quadratic):
            return self.a == other.a and self.b == other.b
        if isinstance(other, (Integral, Fraction)) and not isinstance(other, bool):
            return self.b == 0 and self.a == other
        return NotImplemented

    def __hash__(self):
        return hash(self.a) if self.b == 0 else hash((self.a, self.b))

    def __bool__(self):
        return bool(self.a or self.b)

    def __neg__(self):
        return Quadratic(-self.a, -self.b)

    def __add__(self, other):
        if isinstance(other, Quadratic):
            return Quadratic(self.a + other.a, self.b + other.b)
        return Quadratic(self.a + _rational(other), self.b)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Quadratic):
            return Quadratic(self.a - other.a, self.b - other.b)
        return Quadratic(self.a - _rational(other), self.b)

    def __rsub__(self, other):
        return Quadratic(_rational(other) - self.a, -self.b)

    def __mul__(self, other):
        if isinstance(other, Quadratic):
            return Quadratic(self.a*other.a + 2*self.b*other.b, self.a*other.b + self.b*other.a)
        factor = _rational(other)
        return Quadratic(self.a*factor, self.b*factor)

    __rmul__ = __mul__

    def __truediv__(self, rational):
        divisor = _rational(rational)
        return Quadratic(self.a/divisor, self.b/divisor)

    def sign(self):
        """Compare at √2 > 0, using squares only when the two terms have opposite signs."""
        a_sign = (self.a > 0) - (self.a < 0)
        b_sign = (self.b > 0) - (self.b < 0)
        if not a_sign:
            return b_sign
        if not b_sign or a_sign == b_sign:
            return a_sign
        left = self.a.numerator * self.b.denominator
        right = self.b.numerator * self.a.denominator
        return a_sign if left*left > 2*right*right else b_sign

    def as_fraction(self):
        if self.b:
            raise ValueError("A nonzero radical coefficient prevents rational projection")
        return self.a


def scalar_to_pair(value):
    if not isinstance(value, Quadratic):
        raise TypeError("Expected Quadratic")
    return [scalar_string(value.a), scalar_string(value.b)]


def scalar_from_pair(pair, *, max_scalar_bits=4096, max_scalar_chars=8192):
    if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(v, str) for v in pair):
        raise ValueError("A quadratic scalar requires a JSON array of two canonical rational strings")
    return Quadratic(*(parse_rational(v, canonical=True, max_bits=max_scalar_bits,
                                      max_chars=max_scalar_chars) for v in pair))
