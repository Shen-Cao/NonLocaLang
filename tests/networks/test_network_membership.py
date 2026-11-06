from fractions import Fraction
import random

import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import SourceNetwork
from nonlocalgames.networks.behavior import RationalBehavior
from nonlocalgames.networks.membership import reconstruct_bilocal_chain


def test_reconstruct_random_no_input_bilocal_distributions_with_reversed_sources():
    rng = random.Random(13)

    def distribution(size):
        weights = [rng.randint(1, 7) for _ in range(size)]
        return tuple(Fraction(w, sum(weights)) for w in weights)

    scenario = Scenario((1, 1, 1), (3, 4, 2))
    graphs = [SourceNetwork(3, ((0, 1), (1, 2))),
              SourceNetwork(3, ((2, 1), (1, 0))),
              SourceNetwork(3, ((1, 0), (2, 1)))]
    for _ in range(5):
        pa, pc = distribution(3), distribution(2)
        conditionals = {(a, c): distribution(4) for a in range(3) for c in range(2)}
        behavior = RationalBehavior(scenario, tuple(pa[a]*pc[c]*conditionals[a, c][b]
                                                    for a, b, c in scenario.outputs))
        for network in graphs:
            result = reconstruct_bilocal_chain(network, behavior)
            assert result.status == "compatible" and result.guarantee == "exact_membership"
            assert result.witness.probabilities() == behavior.probabilities
            assert result.witness.network == network


def test_zero_probability_conditionals_and_deterministic_marginals():
    scenario = Scenario((1, 1, 1), (2, 3, 2))
    values = tuple(Fraction(1, 3) if a == 1 and c == 0 else 0 for a, b, c in scenario.outputs)
    behavior = RationalBehavior(scenario, values)
    result = reconstruct_bilocal_chain(SourceNetwork(3, ((2, 1), (1, 0))), behavior)
    assert result.status == "compatible" and result.witness.probabilities() == behavior.probabilities
    assert result.witness.source_probabilities == ((Fraction(1), Fraction(0)),
                                                  (Fraction(0), Fraction(1)))


def test_common_bit_is_exactly_incompatible_with_no_input_bilocality():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    behavior = RationalBehavior(scenario, tuple(Fraction(1, 2) if len(set(a)) == 1 else 0
                                                for a in scenario.outputs))
    result = reconstruct_bilocal_chain(SourceNetwork(3, ((0, 1), (1, 2))), behavior)
    assert result.status == "incompatible" and result.witness is None
    assert result.violation.lhs == Fraction(1, 2)
    assert result.violation.rhs == Fraction(1, 4)


def test_membership_requires_exact_fragment_and_respects_response_budget():
    graph = SourceNetwork(3, ((0, 1), (1, 2)))
    scenario = Scenario((2, 1, 1), (2, 2, 2))
    behavior = RationalBehavior(scenario, (Fraction(1, 8),)*16)
    with pytest.raises(ValueError, match="no external inputs"):
        reconstruct_bilocal_chain(graph, behavior)
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    behavior = RationalBehavior(scenario, (Fraction(1, 8),)*8)
    with pytest.raises(ValueError, match="exactly sources"):
        reconstruct_bilocal_chain(SourceNetwork(3, ((0, 1), (1, 2), (2, 0))), behavior)
    with pytest.raises(ValueError, match="max_entries"):
        reconstruct_bilocal_chain(graph, behavior, max_response_entries=1)
