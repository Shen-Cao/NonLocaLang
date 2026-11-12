"""Exact finite one-round classical-interface game representation."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from numbers import Integral
from operator import index
from typing import Callable, Mapping

Profile = tuple[int, ...]


def rational(value) -> Fraction:
    """Accept exact numbers; require explicit conversion of floating-point data."""
    if isinstance(value, (float, complex)):
        raise TypeError("Use Fraction or a rational string, not floating-point game data")
    result = Fraction(value)
    # Fraction(Rational) may retain bounded integer components from array libraries.
    return Fraction(index(result.numerator), index(result.denominator))


def _sizes(values):
    values = tuple(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
                         for v in values):
        raise ValueError("Alphabet sizes must be positive integers for at least one party")
    return tuple(int(v) for v in values)


def _index(profile, sizes):
    profile = tuple(profile)
    if len(profile) != len(sizes):
        raise ValueError("Profile has the wrong number of parties")
    index = 0
    for v, size in zip(profile, sizes):
        if isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < size:
            raise ValueError("Profile entry outside its alphabet")
        index = index * size + int(v)
    return index


@dataclass(frozen=True)
class Scenario:
    """Party-local input and output cardinalities; one round, no communication."""
    input_sizes: tuple[int, ...]
    output_sizes: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "input_sizes", _sizes(self.input_sizes))
        object.__setattr__(self, "output_sizes", _sizes(self.output_sizes))
        if len(self.input_sizes) != len(self.output_sizes):
            raise ValueError("Input and output alphabets must have the same parties")

    @property
    def parties(self):
        return len(self.input_sizes)

    @property
    def inputs(self):
        return tuple(product(*(range(v) for v in self.input_sizes)))

    @property
    def outputs(self):
        return tuple(product(*(range(v) for v in self.output_sizes)))

    @property
    def entries(self):
        return prod(self.input_sizes) * prod(self.output_sizes)

    def input_index(self, x):
        return _index(x, self.input_sizes)

    def output_index(self, a):
        return _index(a, self.output_sizes)


@dataclass(frozen=True)
class Game:
    """Rational distribution and score table in lexicographic input/output order.

    Payoffs may be signed. Values are winning probabilities only for a Boolean
    predicate, or acceptance probabilities for rewards in [0, 1].
    """
    scenario: Scenario
    distribution: tuple[Fraction, ...]
    payoff: tuple[Fraction, ...]
    name: str = "unnamed"

    def __post_init__(self):
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        mu = tuple(map(rational, self.distribution))
        score = tuple(map(rational, self.payoff))
        if len(mu) != prod(self.scenario.input_sizes) or len(score) != self.scenario.entries:
            raise ValueError("Table sizes do not match the scenario")
        if any(p < 0 for p in mu) or sum(mu) != 1:
            raise ValueError("Input probabilities must be nonnegative and sum exactly to one")
        object.__setattr__(self, "distribution", mu)
        object.__setattr__(self, "payoff", score)

    @classmethod
    def from_rule(cls, scenario: Scenario, rule: Callable, *, distribution: Mapping | None = None,
                  name="unnamed", max_entries=1_000_000):
        """Compile a rule, treating missing distribution keys as promise probability zero."""
        if scenario.entries > max_entries:
            raise ValueError(f"Dense lowering requires {scenario.entries} entries; limit {max_entries}")
        if distribution is None:
            mu = (Fraction(1, prod(scenario.input_sizes)),) * prod(scenario.input_sizes)
        else:
            for x in distribution:
                scenario.input_index(x)
            mu = tuple(distribution.get(x, 0) for x in scenario.inputs)
        return cls(scenario, mu, tuple(rule(x, a) for x in scenario.inputs
                                     for a in scenario.outputs), name)

    @property
    def inputs(self):
        return self.scenario.inputs

    @property
    def outputs(self):
        return self.scenario.outputs

    def probability(self, x):
        return self.distribution[self.scenario.input_index(x)]

    def reward(self, x, a):
        return self.payoff[self.scenario.input_index(x) * prod(self.scenario.output_sizes)
                           + self.scenario.output_index(a)]

    def coefficient(self, x, a):
        return self.probability(x) * self.reward(x, a)

    @property
    def coefficients(self):
        width = prod(self.scenario.output_sizes)
        return tuple(self.distribution[i // width] * v for i, v in enumerate(self.payoff))

    @property
    def is_acceptance_game(self):
        return all(0 <= v <= 1 for v in self.payoff)
