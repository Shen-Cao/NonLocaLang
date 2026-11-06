"""Independent network checks guarding against false compatibility claims."""
from fractions import Fraction
from itertools import permutations, product

import pytest

from nonlocalgames import AnalysisTask, Game, ResourceSpec, Scenario
from nonlocalgames.networks import SourceNetwork
from nonlocalgames.networks.behavior import RationalBehavior
from nonlocalgames.networks.compatibility import check_source_independence, triangle_finner
from nonlocalgames.networks.membership import reconstruct_bilocal_chain


def test_source_independence_checks_groups_beyond_pairwise_marginals():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    even = RationalBehavior(scenario, tuple(Fraction(1, 4) if sum(a) % 2 == 0 else 0
                                            for a in scenario.outputs))
    empty = SourceNetwork(3, ())
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert check_source_independence(empty, even, groups=((left,), (right,))).status == "not_ruled_out"
    assert check_source_independence(empty, even).status == "incompatible"
    result = check_source_independence(empty, even, groups=((2, 0), (1,)))
    assert result.status == "incompatible"
    assert result.violation.lhs != result.violation.rhs


def test_no_input_chain_all_factor_orderings_and_zero_support_rows():
    scenario = Scenario((1, 1, 1), (3, 2, 3))
    pa, pc = (Fraction(0), Fraction(1, 3), Fraction(2, 3)), (Fraction(3, 4), Fraction(1, 4), Fraction(0))
    conditional = {(a, c): (Fraction((a + c) % 3, 3), Fraction(3 - (a + c) % 3, 3))
                   for a in range(3) for c in range(3)}
    behavior = RationalBehavior(scenario, tuple(pa[a]*pc[c]*conditional[a, c][b]
                                                for a, b, c in scenario.outputs))
    for ab, bc in product(((0, 1), (1, 0)), ((1, 2), (2, 1))):
        for edges in permutations((ab, bc)):
            network = SourceNetwork(3, edges)
            result = reconstruct_bilocal_chain(network, behavior)
            assert result.status == "compatible"
            assert result.witness.probabilities() == behavior.probabilities
            assert result.witness.network == network


def test_no_input_chain_can_have_correlated_endpoints_after_conditioning():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    parity = RationalBehavior(scenario, tuple(Fraction(1, 4) if b == (a ^ c) else 0
                                              for a, b, c in scenario.outputs))
    network = SourceNetwork(3, ((0, 1), (1, 2)))
    assert check_source_independence(network, parity).status == "not_ruled_out"
    assert reconstruct_bilocal_chain(network, parity).witness.probabilities() == parity.probabilities
    # Conditioning on b=0 makes a=c; the compatibility test must not impose
    # conditional independence across an observed central measurement outcome.
    assert parity.probabilities[scenario.output_index((0, 0, 0))] == Fraction(1, 4)
    assert parity.probabilities[scenario.output_index((0, 0, 1))] == 0


def test_finner_exact_saturation_and_common_selector_failure():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    triangle = SourceNetwork(3, ((0, 1), (2, 1), (0, 2)))
    deterministic = RationalBehavior(scenario, tuple(int(a == (0, 0, 0)) for a in scenario.outputs))
    assert triangle_finner(triangle, deterministic).status == "not_ruled_out"
    common_bit = RationalBehavior(scenario, tuple(Fraction(1, 2) if a in ((0, 0, 0), (1, 1, 1)) else 0
                                                 for a in scenario.outputs))
    result = triangle_finner(triangle, common_bit)
    assert result.status == "incompatible"
    assert (result.violation.lhs, result.violation.rhs) == (Fraction(1, 4), Fraction(1, 8))


def test_quantum_network_contract_cannot_borrow_unrestricted_backends():
    scenario = Scenario((2, 2), (2, 2))
    game = Game.from_rule(scenario, lambda x, a: (a[0] ^ a[1]) == x[0]*x[1])
    resource = ResourceSpec("quantum_finite", (2, 2), shared_randomness="forbidden",
                             network=SourceNetwork(2, ((0, 1),)))
    task = AnalysisTask(game, resource)
    assert task.capabilities() == ()
    for method in ("local", "ns", "ns_exact", "npa", "xor", "seesaw", "state_sdp"):
        with pytest.raises(NotImplementedError):
            task.solve(method=method)
