"""Exact rational reference strategies with their explicit game conventions."""
from fractions import Fraction

from .algebra import add, identity, scale, subtract
from .budgets import coerce_budget
from .matrix import RationalMatrix
from .scalars import exact_scalar
from .strategy import ExactQuantumStrategy


def _pauli(budget):
    x = RationalMatrix(((0, 1), (1, 0)), budget=budget)
    z = RationalMatrix(((1, 0), (0, -1)), budget=budget)
    y = RationalMatrix(((0, 0), (0, 0)), ((0, -1), (1, 0)), budget=budget)
    return x, y, z


def _povm(observable, budget):
    unit = identity(observable.dimension, budget=budget)
    return (scale(add(unit, observable, budget=budget), Fraction(1, 2), budget=budget),
            scale(subtract(unit, observable, budget=budget), Fraction(1, 2), budget=budget))


def _cat_state(dimension, budget):
    return RationalMatrix(tuple(tuple(Fraction(1, 2) if i in (0, dimension-1) and j in (0, dimension-1)
                                       else 0 for j in range(dimension)) for i in range(dimension)), budget=budget)


def exact_chsh_strategy(parameter=Fraction(408, 985), *, budget=None):
    """Bell state, A=(Z,X), B=(cZ+sX,cZ-sX), with rational c^2+s^2=1.

    c=(1-t^2)/(1+t^2), s=2t/(1+t^2). At default t=408/985 the uniform CHSH
    score is exactly 3880899/4546756, without claiming exact optimality.
    """
    budget = coerce_budget(budget)
    t = exact_scalar(parameter, max_bits=budget.max_scalar_bits)
    c, s = (1-t*t)/(1+t*t), 2*t/(1+t*t)
    x, _, z = _pauli(budget)
    cz, sx = scale(z, c, budget=budget), scale(x, s, budget=budget)
    observables = ((z, x), (add(cz, sx, budget=budget), subtract(cz, sx, budget=budget)))
    measurements = tuple(tuple(_povm(o, budget) for o in party) for party in observables)
    return ExactQuantumStrategy(_cat_state(4, budget), measurements, (2, 2), budget=budget)


def exact_ghz_strategy(*, budget=None):
    """Three-qubit GHZ density and X/Y projectors; the GHZ promise score is one."""
    budget = coerce_budget(budget)
    x, y, _ = _pauli(budget)
    party = (_povm(x, budget), _povm(y, budget))
    return ExactQuantumStrategy(_cat_state(8, budget), (party,)*3, (2, 2, 2), budget=budget)
