"""Exact maximally entangled witnesses for binary-output game families."""
from fractions import Fraction

from ..quantum.exact import ExactQuantumStrategy, RationalMatrix
from ..quantum.exact.algebra import add, identity, scale, subtract


def bell_strategy(alice, bob, *, budget=None):
    """Measure the normalized sum of |j,j> with the supplied ±1 observables."""
    dimension = alice[0].dimension
    size = dimension**2
    occupied = range(0, size, dimension + 1)
    density = RationalMatrix(tuple(tuple(Fraction(1, dimension) if i in occupied and j in occupied
                                         else 0 for j in range(size)) for i in range(size)),
                             budget=budget)
    unit = identity(dimension, budget=budget)
    half = Fraction(1, 2)

    def measurements(observables):
        return tuple((scale(add(unit, op, budget=budget), half, budget=budget),
                      scale(subtract(unit, op, budget=budget), half, budget=budget))
                     for op in observables)

    return ExactQuantumStrategy(density, (measurements(alice), measurements(bob)), budget=budget)


def planar_bell_strategy(alice, bob, *, budget=None):
    """A real unit Bloch vector (c,s) specifies the observable cZ+sX."""
    def observables(vectors):
        return tuple(RationalMatrix(((c, s), (s, -c)), budget=budget) for c, s in vectors)

    return bell_strategy(observables(alice), observables(bob), budget=budget)
