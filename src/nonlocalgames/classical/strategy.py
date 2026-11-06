"""Validated deterministic local witnesses with exact scoring."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from numbers import Integral

from ..core.game import Scenario


def response_tables(scenario, responses):
    """Normalize party/input response tables without accepting rounded labels."""
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    responses = tuple(tuple(table) for table in responses)
    if len(responses) != scenario.parties:
        raise ValueError("Provide one deterministic response table per party")
    for table, inputs, outputs in zip(responses, scenario.input_sizes, scenario.output_sizes):
        if len(table) != inputs or any(isinstance(a, bool) or not isinstance(a, Integral)
                                       or not 0 <= a < outputs for a in table):
            raise ValueError("Response tables must have one valid integer output per local input")
    return tuple(tuple(int(a) for a in table) for table in responses)


@dataclass(frozen=True)
class DeterministicStrategy:
    """A local response function for each party, with no randomness required.

    The same tables are admissible whether external shared randomness is
    allowed or forbidden. Their exact score does not assert optimality.
    """
    scenario: Scenario
    responses: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "responses", response_tables(self.scenario, self.responses))

    def probabilities(self, scenario=None, *, max_entries=1_000_000):
        """Return the exact deterministic behavior in input-major/output-minor order."""
        if scenario is not None and scenario != self.scenario:
            raise ValueError("Witness and requested scenario differ")
        if isinstance(max_entries, bool) or not isinstance(max_entries, Integral) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if self.scenario.entries > max_entries:
            raise ValueError("Deterministic behavior expansion exceeds max_entries")
        width = prod(self.scenario.output_sizes)
        result = []
        for x in product(*(range(n) for n in self.scenario.input_sizes)):
            block = [Fraction(0)] * width
            output = tuple(table[q] for table, q in zip(self.responses, x))
            block[self.scenario.output_index(output)] = Fraction(1)
            result.extend(block)
        return tuple(result)

    def score(self, game):
        """Evaluate a matching rational verifier without expanding a behavior table."""
        if game.scenario != self.scenario:
            raise ValueError("Witness and game scenarios differ")
        return sum((game.coefficient(x, tuple(table[q] for table, q in zip(self.responses, x)))
                    for x in product(*(range(n) for n in self.scenario.input_sizes))), Fraction(0))


def deterministic_quantum_strategy(scenario, responses):
    """Embed deterministic tables in one-dimensional product quantum systems.

    Identity/zero effects are projective. This witness satisfies every positive
    dimension upper bound, every PPT cut, and a product-state restriction. It
    does not implement a prescribed fixed state with different tensor factors.
    """
    import numpy as np
    from ..quantum.strategy import QuantumStrategy
    tables = response_tables(scenario, responses)
    measurements = tuple(tuple(tuple(np.array([[int(a == tables[i][x])]], dtype=complex)
                                      for a in range(scenario.output_sizes[i]))
                                for x in range(scenario.input_sizes[i]))
                         for i in range(scenario.parties))
    return QuantumStrategy(np.ones((1, 1)), measurements)
