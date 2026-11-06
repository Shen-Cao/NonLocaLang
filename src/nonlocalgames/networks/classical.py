"""Exact evaluation of finite classical independent-source witnesses."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from numbers import Integral

from ..core.game import Scenario, rational
from .structure import SourceNetwork


@dataclass(frozen=True)
class ClassicalNetworkStrategy:
    """Independent source distributions and local conditional response tables.

    ``responses[i]`` is flat: local input first, then the tuple of incident
    source values in increasing source-index order, then local output. Private
    local randomness is represented by stochastic response rows. No joint
    distribution over different sources can be supplied to this constructor.
    """
    network: SourceNetwork
    scenario: Scenario
    source_probabilities: tuple[tuple[Fraction, ...], ...]
    responses: tuple[tuple[Fraction, ...], ...]

    def __post_init__(self):
        if not isinstance(self.network, SourceNetwork) or not isinstance(self.scenario, Scenario):
            raise TypeError("Expected a SourceNetwork and Scenario")
        if self.network.parties != self.scenario.parties:
            raise ValueError("Network and scenario must have the same parties")
        sources = tuple(tuple(map(rational, p)) for p in self.source_probabilities)
        responses = tuple(tuple(map(rational, p)) for p in self.responses)
        if len(sources) != len(self.network.sources):
            raise ValueError("Provide one independent probability vector per source")
        if any(not p or any(v < 0 for v in p) or sum(p) != 1 for p in sources):
            raise ValueError("Each source distribution must be nonnegative and normalized")
        if len(responses) != self.network.parties:
            raise ValueError("Provide one local response table per party")
        for i, table in enumerate(responses):
            width = self.scenario.output_sizes[i]
            rows = self.scenario.input_sizes[i] * prod(len(sources[s]) for s in self.network.incident_sources(i))
            if len(table) != rows * width:
                raise ValueError("Local response table does not match input/source/output sizes")
            if any(v < 0 for v in table) or any(sum(table[j:j+width]) != 1 for j in range(0, len(table), width)):
                raise ValueError("Each local response row must be nonnegative and normalized")
        object.__setattr__(self, "source_probabilities", sources)
        object.__setattr__(self, "responses", responses)

    @classmethod
    def from_response_rules(cls, network, scenario, source_probabilities, rules, *, max_entries=1_000_000):
        """Compile rules ``rules[i](input_i, incident_source_values, output_i)``."""
        sources, rules = tuple(tuple(p) for p in source_probabilities), tuple(rules)
        if not isinstance(network, SourceNetwork) or not isinstance(scenario, Scenario):
            raise TypeError("Expected a SourceNetwork and Scenario")
        if network.parties != scenario.parties or len(rules) != network.parties:
            raise ValueError("Provide matching parties and one response rule per party")
        if len(sources) != len(network.sources) or any(not p for p in sources):
            raise ValueError("Provide one nonempty distribution per source")
        if isinstance(max_entries, bool) or not isinstance(max_entries, Integral) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        sizes = [scenario.input_sizes[i] * scenario.output_sizes[i] *
                 prod(len(sources[s]) for s in network.incident_sources(i)) for i in range(network.parties)]
        if sum(sizes) > max_entries:
            raise ValueError("Response table expansion exceeds max_entries")
        tables = []
        for i, rule in enumerate(rules):
            latent = tuple(product(*(range(len(sources[s])) for s in network.incident_sources(i))))
            tables.append(tuple(rule(x, values, a) for x in range(scenario.input_sizes[i])
                                for values in latent for a in range(scenario.output_sizes[i])))
        return cls(network, scenario, sources, tuple(tables))

    def probabilities(self, scenario=None, *, max_evaluations=10_000_000):
        """Return exact input-major probabilities; keep source factors independent."""
        if scenario is not None and scenario != self.scenario:
            raise ValueError("Strategy and requested scenario differ")
        if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer")
        latent_count = prod(len(p) for p in self.source_probabilities)
        if latent_count * self.scenario.entries > max_evaluations:
            raise ValueError("Network evaluation exceeds max_evaluations")
        inputs, outputs = self.scenario.inputs, self.scenario.outputs
        behavior = [Fraction(0) for _ in range(self.scenario.entries)]
        incident = tuple(self.network.incident_sources(i) for i in range(self.network.parties))
        for latent in product(*(range(len(p)) for p in self.source_probabilities)):
            weight = prod((p[k] for p, k in zip(self.source_probabilities, latent)), start=Fraction(1))
            for xi, x in enumerate(inputs):
                offsets = []
                for i in range(self.network.parties):
                    row = x[i]
                    for source in incident[i]:
                        row = row * len(self.source_probabilities[source]) + latent[source]
                    offsets.append(row * self.scenario.output_sizes[i])
                for ai, a in enumerate(outputs):
                    behavior[xi * len(outputs) + ai] += weight * prod(
                        (self.responses[i][offsets[i] + a[i]] for i in range(self.network.parties)),
                        start=Fraction(1))
        return tuple(behavior)

    def score(self, game):
        if game.scenario != self.scenario:
            raise ValueError("Strategy and game scenarios differ")
        return sum((c*p for c, p in zip(game.coefficients, self.probabilities())), Fraction(0))


def deterministic_network_strategy(network, scenario, tables, *, source_probabilities=None):
    """Lift an ordinary local deterministic strategy by ignoring every source.

    Every independent-source graph admits this witness. Therefore the supremum
    of any unconstrained linear game score over classical network strategies
    equals its ordinary local value: the network set is a subset of the local
    set and contains all deterministic local behaviors. This does not equate
    the two behavior sets or solve a network distribution-compatibility problem.
    """
    if not isinstance(network, SourceNetwork) or not isinstance(scenario, Scenario):
        raise TypeError("Expected a SourceNetwork and Scenario")
    tables = tuple(tuple(table) for table in tables)
    if network.parties != scenario.parties or len(tables) != network.parties:
        raise ValueError("Provide one deterministic response table per party")
    for i, table in enumerate(tables):
        if len(table) != scenario.input_sizes[i] or any(isinstance(a, bool) or not isinstance(a, Integral)
                                                      or not 0 <= a < scenario.output_sizes[i] for a in table):
            raise ValueError("Deterministic response table has invalid inputs or outputs")
    sources = ((1,),) * len(network.sources) if source_probabilities is None else source_probabilities
    rules = []
    for table in tables:
        def rule(x, values, a, table=table):
            return a == table[x]
        rules.append(rule)
    return ClassicalNetworkStrategy.from_response_rules(network, scenario, sources, rules)
