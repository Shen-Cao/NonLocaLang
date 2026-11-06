"""Constructive exact membership for a restricted no-input bilocal fragment."""
from dataclasses import dataclass
from fractions import Fraction

from .behavior import RationalBehavior, positive_budget
from .classical import ClassicalNetworkStrategy
from .compatibility import CompatibilityViolation, check_source_independence
from .structure import SourceNetwork


@dataclass(frozen=True)
class BilocalMembershipResult:
    """Exact decision for an unrestricted classical no-input A--B--C chain.

    Compatibility comes with an explicit finite independent-source witness.
    This result does not impose source cardinality or quantum dimension bounds.
    """
    network: SourceNetwork
    status: str
    witness: ClassicalNetworkStrategy | None
    violation: CompatibilityViolation | None
    guarantee: str = "exact_membership"
    resource: str = "classical_independent_sources"


def reconstruct_bilocal_chain(network, behavior, *, max_response_entries=1_000_000):
    """Decide and reconstruct no-input bilocality via P_AC=P_A P_C.

    Parties are A=0, B=1, C=2. The graph must have exactly the independent
    sources AB and BC, in either source/endpoint order. Source AB distributes
    a sampled A output and source BC distributes a sampled C output; Bob uses
    the conditional distribution P(b|a,c). Zero-support conditional rows are
    arbitrary and implemented as deterministic output zero. See Renou et al.,
    Eq. (1) and its constructive discussion: https://arxiv.org/abs/1901.08287.
    """
    if not isinstance(network, SourceNetwork) or not isinstance(behavior, RationalBehavior):
        raise TypeError("Expected a SourceNetwork and exactly validated RationalBehavior")
    if network.parties != 3 or behavior.scenario.parties != 3:
        raise ValueError("Bilocal reconstruction requires exactly three parties")
    if len(network.sources) != 2 or {frozenset(s) for s in network.sources} != {
            frozenset((0, 1)), frozenset((1, 2))}:
        raise ValueError("Bilocal reconstruction requires exactly sources AB and BC")
    if behavior.scenario.input_sizes != (1, 1, 1):
        raise ValueError("Constructive bilocal membership currently requires no external inputs")
    positive_budget(max_response_entries, "max_response_entries")
    checked = check_source_independence(network, behavior, groups=((0,), (2,)))
    if checked.violation is not None:
        return BilocalMembershipResult(network, "incompatible", None, checked.violation)
    scenario, x = behavior.scenario, (0, 0, 0)
    marginal_a, marginal_c = behavior.marginal((0,), x), behavior.marginal((2,), x)
    source_ab = next(s for s, edge in enumerate(network.sources) if 0 in edge)
    source_bc = next(s for s, edge in enumerate(network.sources) if 2 in edge)
    sources = []
    for s in range(2):
        marginal = marginal_a if s == source_ab else marginal_c
        sources.append(tuple(marginal[(a,)] for a in range(len(marginal))))
    bob_positions = {s: i for i, s in enumerate(network.incident_sources(1))}

    def outer(local_input, latent, output):
        return output == latent[0]

    def bob(local_input, latent, b):
        a, c = latent[bob_positions[source_ab]], latent[bob_positions[source_bc]]
        denominator = marginal_a[(a,)] * marginal_c[(c,)]
        if denominator == 0:
            return Fraction(b == 0)
        return behavior.probabilities[scenario.output_index((a, b, c))] / denominator

    witness = ClassicalNetworkStrategy.from_response_rules(
        network, scenario, sources, (outer, bob, outer), max_entries=max_response_entries)
    return BilocalMembershipResult(network, "compatible", witness, None)
