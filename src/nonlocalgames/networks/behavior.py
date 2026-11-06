"""Exact finite behaviors with checked normalization and nonsignalling."""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from numbers import Integral

from ..core.game import Scenario, rational


def positive_budget(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def party_subset(parties, count, *, allow_empty=True):
    parties = tuple(parties)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < count for i in parties):
        raise ValueError("Party subset indices must be integers within the scenario")
    if len(set(parties)) != len(parties) or (not allow_empty and not parties):
        raise ValueError("Party subsets must have distinct entries and be nonempty when required")
    return tuple(map(int, parties))


@dataclass(frozen=True)
class RationalBehavior:
    """An exactly normalized nonsignalling conditional distribution.

    Input-major/output-minor table entries must be integers, rational strings,
    or Fractions. Floating data is rejected, not reconstructed automatically.
    A valid behavior is not necessarily local, quantum, or network-compatible.
    """
    scenario: Scenario
    probabilities: tuple[Fraction, ...]
    max_entries: int = field(default=1_000_000, kw_only=True, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        budget = positive_budget(self.max_entries, "max_entries")
        if self.scenario.entries > budget:
            raise ValueError("Behavior table exceeds max_entries")
        values = tuple(map(rational, self.probabilities))
        if len(values) != self.scenario.entries:
            raise ValueError("Behavior table does not match scenario entries")
        if any(p < 0 for p in values):
            raise ValueError("Behavior probabilities must be nonnegative")
        width = len(self.scenario.outputs)
        if any(sum(values[start:start+width]) != 1 for start in range(0, len(values), width)):
            raise ValueError("Behavior probabilities must sum exactly to one at every input")
        object.__setattr__(self, "probabilities", values)
        self._validate_nonsignalling()

    def marginal(self, parties, full_input):
        """Return output-profile probabilities in the supplied party-subset order."""
        parties = party_subset(parties, self.scenario.parties)
        xi = self.scenario.input_index(full_input)
        outputs = self.scenario.outputs
        result = {a: Fraction(0) for a in product(*(range(self.scenario.output_sizes[i]) for i in parties))}
        for ai, a in enumerate(outputs):
            result[tuple(a[i] for i in parties)] += self.probabilities[xi * len(outputs) + ai]
        return result

    def _validate_nonsignalling(self):
        # Complement marginals independent of each omitted party imply all subset conditions.
        for party in range(self.scenario.parties):
            complement = tuple(i for i in range(self.scenario.parties) if i != party)
            references = {}
            for x in self.scenario.inputs:
                local_input = tuple(x[i] for i in complement)
                marginal = self.marginal(complement, x)
                if local_input in references and references[local_input] != marginal:
                    raise ValueError(f"Behavior violates exact nonsignalling for omitted party {party}")
                references[local_input] = marginal
