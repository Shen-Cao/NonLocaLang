"""Exact necessary conditions for independent-source compatibility.

These checks can rule a network out. Passing does not certify membership in
any classical or quantum network correlation set.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod

from .behavior import RationalBehavior, party_subset, positive_budget
from .structure import SourceNetwork


@dataclass(frozen=True)
class CompatibilityViolation:
    input_profile: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    group_outputs: tuple[tuple[int, ...], ...]
    lhs: Fraction
    rhs: Fraction
    required_relation: str


@dataclass(frozen=True)
class CompatibilityResult:
    """A counterexample or an explicitly inconclusive necessary-condition result."""
    method: str
    network: SourceNetwork
    status: str
    checks: int
    violation: CompatibilityViolation | None
    assumptions: tuple[str, ...]
    guarantee: str = "exact_necessary_condition"
    applicable_resources: tuple[str, ...] = ("classical_independent_sources", "quantum_independent_sources")


def _validate_context(network, behavior):
    if not isinstance(network, SourceNetwork) or not isinstance(behavior, RationalBehavior):
        raise TypeError("Expected a SourceNetwork and exactly validated RationalBehavior")
    if network.parties != behavior.scenario.parties:
        raise ValueError("Network and behavior have different parties")


def _groups(network, groups):
    groups = tuple(party_subset(group, network.parties, allow_empty=False) for group in groups)
    if len(groups) < 2:
        raise ValueError("Specify at least two nonempty party groups")
    flat = tuple(i for group in groups for i in group)
    if len(set(flat)) != len(flat):
        raise ValueError("Party groups must be disjoint")
    used_sources = set()
    for group in groups:
        incident = {s for party in group for s in network.incident_sources(party)}
        if used_sources & incident:
            raise ValueError("The specified groups share an incident source")
        used_sources |= incident
    return groups


def source_disjoint_pairs(network, *, max_group_pairs=10_000):
    """Enumerate all unordered nonempty disjoint party-subset pairs with disjoint sources.

    The guard limits all candidate subset pairs before filtering by incidence.
    Callers can select groups explicitly to avoid this exponential enumeration.
    """
    if not isinstance(network, SourceNetwork):
        raise TypeError("network must be a SourceNetwork")
    budget = positive_budget(max_group_pairs, "max_group_pairs")
    candidates = (3**network.parties - 2*2**network.parties + 1) // 2
    if candidates > budget:
        raise ValueError("Party-subset enumeration exceeds max_group_pairs; specify groups")
    pairs = []
    for assignment in product(range(3), repeat=network.parties):
        first = tuple(i for i, role in enumerate(assignment) if role == 1)
        second = tuple(i for i, role in enumerate(assignment) if role == 2)
        if not first or not second or first > second:
            continue
        first_sources = {s for i in first for s in network.incident_sources(i)}
        second_sources = {s for i in second for s in network.incident_sources(i)}
        if first_sources.isdisjoint(second_sources):
            pairs.append((first, second))
    return tuple(pairs)


def check_source_independence(network, behavior, *, groups=None, max_group_pairs=10_000,
                              max_checks=1_000_000):
    """Test marginal factorization for groups with no common incident sources.

    With ``groups=None`` all source-disjoint subset pairs are tested. Explicit
    groups may contain two or more subsets. No outputs are conditioned upon:
    conditioning on an intermediate node can create correlations and is not
    covered by this test. External inputs are independent of all sources.
    """
    _validate_context(network, behavior)
    positive_budget(max_checks, "max_checks")
    families = source_disjoint_pairs(network, max_group_pairs=max_group_pairs) if groups is None else (_groups(network, groups),)
    scenario = behavior.scenario
    estimated = len(scenario.inputs) * sum(prod(scenario.output_sizes[i] for g in family for i in g)
                                          for family in families)
    if estimated > max_checks:
        raise ValueError("Marginal factorization checks exceed max_checks")
    checks = 0
    assumptions = ("independent sources with the declared incidence",
                   "local operations without communication or extra global randomness",
                   "source-independent external inputs; no conditioning on other outputs")
    for family in families:
        union = tuple(sorted(i for group in family for i in group))
        for x in scenario.inputs:
            marginals = tuple(behavior.marginal(group, x) for group in family)
            joint = behavior.marginal(union, x)
            for group_outputs in product(*(tuple(marginal) for marginal in marginals)):
                labels = {i: a for group, values in zip(family, group_outputs) for i, a in zip(group, values)}
                lhs = joint[tuple(labels[i] for i in union)]
                rhs = prod((m[a] for m, a in zip(marginals, group_outputs)), start=Fraction(1))
                checks += 1
                if lhs != rhs:
                    violation = CompatibilityViolation(x, family, group_outputs, lhs, rhs, "equal")
                    return CompatibilityResult("source_disjoint_independence", network, "incompatible", checks, violation, assumptions)
    return CompatibilityResult("source_disjoint_independence", network, "not_ruled_out", checks, None, assumptions)


def triangle_finner(network, behavior, *, max_checks=1_000_000):
    """Check P(abc|x)^2 <= P_A(a|x_A) P_B(b|x_B) P_C(c|x_C).

    The graph must consist of exactly one independent source per pair of three
    parties. This necessary condition holds for classical and quantum sources;
    see Renou et al., Theorem 1, https://arxiv.org/abs/1901.08287. We make no
    assertion for arbitrary generalized no-signalling source theories.
    """
    _validate_context(network, behavior)
    positive_budget(max_checks, "max_checks")
    validate_triangle(network)
    if behavior.scenario.entries > max_checks:
        raise ValueError("Triangle event checks exceed max_checks")
    assumptions = ("three independent bipartite classical or quantum sources in a triangle",
                   "local measurements without communication or extra global randomness",
                   "external inputs independent of sources")
    groups, checks = ((0,), (1,), (2,)), 0
    outputs = behavior.scenario.outputs
    for xi, x in enumerate(behavior.scenario.inputs):
        marginals = tuple(behavior.marginal(group, x) for group in groups)
        for ai, a in enumerate(outputs):
            group_outputs = tuple((value,) for value in a)
            lhs = behavior.probabilities[xi*len(outputs)+ai]**2
            rhs = prod((m[output] for m, output in zip(marginals, group_outputs)), start=Fraction(1))
            checks += 1
            if lhs > rhs:
                violation = CompatibilityViolation(x, groups, group_outputs, lhs, rhs, "less_or_equal")
                return CompatibilityResult("triangle_finner", network, "incompatible", checks, violation, assumptions)
    return CompatibilityResult("triangle_finner", network, "not_ruled_out", checks, None, assumptions)


def validate_triangle(network):
    """Require the exact three-pair-source triangle supported by this checker."""
    if not isinstance(network, SourceNetwork):
        raise TypeError("network must be a SourceNetwork")
    edges = {frozenset(edge) for edge in network.sources}
    if network.parties != 3 or len(network.sources) != 3 or edges != {
            frozenset((0, 1)), frozenset((1, 2)), frozenset((0, 2))}:
        raise ValueError("Triangle requires exactly one independent source on each party pair")
