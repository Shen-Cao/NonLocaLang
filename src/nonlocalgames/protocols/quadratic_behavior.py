"""Normalized nonsignalling conditional tables over Q(√2) and their wiring targets."""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import islice
from math import prod

from ..core.game import Scenario
from ..core.numeric_data import positive_budget
from ..core.scalar_wire import ScalarBudgetExceeded
from ..networks.behavior import RationalBehavior, party_subset
from .quadratic import Quadratic
from .wiring import (DEFAULT_MAX_PARTIES, DEFAULT_MAX_TOTAL_ENTRIES, WiringArena,
                     WiringBudgetExceeded, WiringTarget, _aligned_behaviors)


def _bound(bits, *components):
    if any(c.numerator.bit_length() > bits or c.denominator.bit_length() > bits for c in components):
        raise ScalarBudgetExceeded("Quadratic coefficient exceeds max_scalar_bits")


def _sum(values, bits):
    a = b = Fraction(0)
    for value in values:
        a += value.a
        b += value.b
        _bound(bits, a, b)
    return Quadratic(a, b)


def _check_nonsignalling(scenario, values, width, bits):
    """Compare complement marginals to those at the omitted party's question zero."""
    for i, (nx, na) in enumerate(zip(scenario.input_sizes, scenario.output_sizes)):
        if nx == 1 or scenario.parties == 1:
            continue
        x_stride = prod(scenario.input_sizes[i+1:])
        a_stride = prod(scenario.output_sizes[i+1:])
        for xi in range(len(values)//width):
            question = (xi//x_stride) % nx
            if question == 0:
                continue
            ref = (xi-question*x_stride)*width
            for block in range(0, width, na*a_stride):
                for suffix in range(a_stride):
                    indices = range(block+suffix, block+suffix+na*a_stride, a_stride)
                    observed = _sum((values[xi*width+j] for j in indices), bits)
                    expected = _sum((values[ref+j] for j in indices), bits)
                    if observed != expected:
                        raise ValueError(f"Quadratic behavior signals from party {i}")


@dataclass(frozen=True, init=False)
class QuadraticBehavior:
    """An input-major, output-minor table with validated real probabilities.

    The scalar budget covers supplied coefficients and running marginal sums.
    Validation scans O(parties × entries) terms, with exact rational arithmetic.
    """
    scenario: Scenario
    probabilities: tuple[Quadratic, ...]
    max_scalar_bits: int = field(repr=False, compare=False)

    def __init__(self, scenario, probabilities, *, max_entries=DEFAULT_MAX_TOTAL_ENTRIES,
                 max_scalar_bits=4096, max_parties=DEFAULT_MAX_PARTIES):
        if not isinstance(scenario, Scenario):
            raise TypeError("Expected a Scenario")
        limit = positive_budget(max_entries, "max_entries")
        bits = positive_budget(max_scalar_bits, "max_scalar_bits")
        if scenario.parties > positive_budget(max_parties, "max_parties"):
            raise WiringBudgetExceeded("Quadratic behavior exceeds max_parties")
        entries = 1
        for size in scenario.input_sizes + scenario.output_sizes:
            if size > limit // entries:
                raise WiringBudgetExceeded("Quadratic behavior exceeds max_entries")
            entries *= size
        values = tuple(islice(iter(probabilities), entries+1))
        if len(values) != entries:
            raise ValueError("Quadratic table length differs from the scenario")
        for value in values:
            if not isinstance(value, Quadratic):
                raise TypeError("Behavior entries must be Quadratic scalars")
            _bound(bits, value.a, value.b)
            if value.sign() < 0:
                raise ValueError("Quadratic probabilities must be nonnegative")
        width = prod(scenario.output_sizes)
        for start in range(0, entries, width):
            if _sum(values[start:start+width], bits) != 1:
                raise ValueError("Every quadratic probability row must sum exactly to one")
        _check_nonsignalling(scenario, values, width, bits)
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "max_scalar_bits", bits)

    def marginal(self, parties, full_input):
        scenario = self.scenario
        parties = party_subset(islice(iter(parties), scenario.parties+1), scenario.parties)
        xi = scenario.input_index(islice(iter(full_input), scenario.parties+1))
        start = xi * prod(scenario.output_sizes)
        result = {}
        for j, output in enumerate(scenario.outputs):
            key = tuple(output[i] for i in parties)
            subtotal = result.get(key, 0) + self.probabilities[start+j]
            _bound(self.max_scalar_bits, subtotal.a, subtotal.b)
            result[key] = subtotal
        return result

    def as_rational(self):
        return RationalBehavior(self.scenario, tuple(p.as_fraction() for p in self.probabilities),
                                max_entries=len(self.probabilities))


@dataclass(frozen=True, init=False)
class QuadraticWiringTarget:
    arena: WiringArena
    behaviors: tuple[QuadraticBehavior, ...]

    def __init__(self, arena, by_route):
        behaviors = _aligned_behaviors(arena, by_route, QuadraticBehavior)
        object.__setattr__(self, "arena", arena)
        object.__setattr__(self, "behaviors", behaviors)

    def as_rational(self):
        return WiringTarget(self.arena, {r: b.as_rational() for r, b in zip(self.arena.routes, self.behaviors)})
