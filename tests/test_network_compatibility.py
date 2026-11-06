from fractions import Fraction
import random

import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import ClassicalNetworkStrategy, SourceNetwork, triangle_xor_strategy
from nonlocalgames.networks.behavior import RationalBehavior
from nonlocalgames.networks.compatibility import (check_source_independence, source_disjoint_pairs,
                                                 triangle_finner)


def common_bit(scenario):
    return RationalBehavior(scenario, tuple(Fraction(1, 2) if len(set(a)) == 1 else 0
                                           for x in scenario.inputs for a in scenario.outputs))


def test_common_bit_rules_out_chain_and_triangle_exactly():
    behavior = common_bit(Scenario((1, 1, 1), (2, 2, 2)))
    chain = SourceNetwork(3, ((0, 1), (1, 2)))
    result = check_source_independence(chain, behavior)
    assert result.status == "incompatible"
    assert result.violation.lhs == Fraction(1, 2)
    assert result.violation.rhs == Fraction(1, 4)
    assert result.violation.groups == ((0,), (2,))
    triangle = SourceNetwork(3, ((2, 0), (1, 0), (2, 1)))
    result = triangle_finner(triangle, behavior)
    assert result.status == "incompatible"
    assert result.violation.lhs == Fraction(1, 4)
    assert result.violation.rhs == Fraction(1, 8)
    assert result.violation.group_outputs == ((0,), (0,), (0,))
    assert result.guarantee == "exact_necessary_condition"


def test_valid_triangle_xor_passes_without_claiming_membership():
    strategy = triangle_xor_strategy()
    behavior = RationalBehavior(strategy.scenario, strategy.probabilities())
    result = triangle_finner(strategy.network, behavior)
    assert result.status == "not_ruled_out" and result.violation is None
    assert result.checks == 8
    independence = check_source_independence(strategy.network, behavior)
    assert independence.checks == 0 and independence.status == "not_ruled_out"


def test_triangle_finner_passes_w_distribution_inconclusively():
    # A deliberately weak test: no membership assertion is made for passed inequalities.
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    behavior = RationalBehavior(scenario, tuple(Fraction(1, 3) if sum(a) == 1 else 0 for a in scenario.outputs))
    triangle = SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))
    result = triangle_finner(triangle, behavior)
    assert result.status == "not_ruled_out"


def test_random_rational_classical_triangle_strategies_pass_all_inputs():
    rng = random.Random(8)
    graph, scenario = SourceNetwork(3, ((0, 1), (1, 2), (2, 0))), Scenario((2, 2, 1), (2, 3, 2))

    def random_distribution(size):
        weights = [rng.randint(0, 8) for _ in range(size)]
        if not sum(weights):
            weights[0] = 1
        return tuple(Fraction(w, sum(weights)) for w in weights)

    for _ in range(5):
        sources = tuple(random_distribution(2) for _ in range(3))
        responses = tuple(tuple(v for _ in range(nx*4) for v in random_distribution(na))
                          for nx, na in zip(scenario.input_sizes, scenario.output_sizes))
        strategy = ClassicalNetworkStrategy(graph, scenario, sources, responses)
        result = triangle_finner(graph, RationalBehavior(scenario, strategy.probabilities()))
        assert result.status == "not_ruled_out" and result.checks == scenario.entries


def test_explicit_groups_and_private_randomness_factorization():
    graph, scenario = SourceNetwork(4, ((0, 1), (2,))), Scenario((1, 1, 1, 1), (2, 2, 2, 2))
    probabilities = tuple(Fraction(1, 8) if a[0] == a[1] else 0 for a in scenario.outputs)
    behavior = RationalBehavior(scenario, probabilities)
    result = check_source_independence(graph, behavior, groups=((1, 0), (2,), (3,)))
    assert result.status == "not_ruled_out" and result.checks == 16
    assert check_source_independence(graph, behavior).status == "not_ruled_out"
    with pytest.raises(ValueError, match="share an incident"):
        check_source_independence(graph, behavior, groups=((0,), (1,)))
    for groups in [((0,), (0, 1)), ((), (2,)), ((0,),), ((4,), (0,))]:
        with pytest.raises(ValueError):
            check_source_independence(graph, behavior, groups=groups)


def test_behavior_rejects_float_unnormalized_and_signalling_tables():
    scenario = Scenario((2, 2), (2, 2))
    with pytest.raises(TypeError):
        RationalBehavior(scenario, [0.25]*16)
    with pytest.raises(ValueError, match="sum exactly"):
        RationalBehavior(scenario, [0]*16)
    signalling = tuple(int(a == (x[1], 0)) for x in scenario.inputs for a in scenario.outputs)
    with pytest.raises(ValueError, match="nonsignalling"):
        RationalBehavior(scenario, signalling)
    with pytest.raises(ValueError):
        RationalBehavior(scenario, [Fraction(1, 4)]*16, max_entries=15)


def test_no_signalling_checked_for_joint_complement_not_only_single_marginals():
    scenario = Scenario((1, 1, 2), (2, 2, 1))
    values = tuple(Fraction(1, 2) if a[0] ^ a[1] == x[2] else 0
                   for x in scenario.inputs for a in scenario.outputs)
    with pytest.raises(ValueError, match="nonsignalling"):
        RationalBehavior(scenario, values)


def test_wrong_graph_and_expansion_guards():
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    behavior = common_bit(scenario)
    for graph in [SourceNetwork(3, ((0, 1), (1, 2))),
                  SourceNetwork(3, ((0, 1), (1, 2), (0, 1))),
                  SourceNetwork(3, ((0, 1), (1, 2), (0, 1, 2)))]:
        with pytest.raises(ValueError):
            triangle_finner(graph, behavior)
    with pytest.raises(ValueError, match="different parties"):
        check_source_independence(SourceNetwork(2, ()), behavior)
    with pytest.raises(ValueError, match="max_group_pairs"):
        source_disjoint_pairs(SourceNetwork(8, ()), max_group_pairs=10)
    with pytest.raises(ValueError, match="max_checks"):
        check_source_independence(SourceNetwork(3, ()), behavior, max_checks=1)
    with pytest.raises(ValueError, match="max_checks"):
        triangle_finner(SourceNetwork(3, ((0, 1), (1, 2), (2, 0))), behavior, max_checks=1)


def test_serialized_exact_counterexamples_recheck_and_reject_corruption():
    import json
    from dataclasses import replace
    from nonlocalgames.networks.behavior_serialization import behavior_from_dict, behavior_to_dict
    from nonlocalgames.networks.compatibility_certificates import (verify_compatibility_violation,
                                                                  violation_to_dict)
    from nonlocalgames.networks.serialization import network_from_dict, network_to_dict
    behavior = common_bit(Scenario((1, 1, 1), (2, 2, 2)))
    for graph, checker in [(SourceNetwork(3, ((0, 1), (1, 2))), check_source_independence),
                           (SourceNetwork(3, ((0, 1), (1, 2), (2, 0))), triangle_finner)]:
        result = checker(graph, behavior)
        payload = json.loads(json.dumps({"network": network_to_dict(graph),
                                         "behavior": behavior_to_dict(behavior),
                                         "violation": violation_to_dict(result.violation)}))
        loaded_graph = network_from_dict(payload["network"])
        loaded_behavior = behavior_from_dict(payload["behavior"])
        rechecked = verify_compatibility_violation(loaded_graph, loaded_behavior, result.method, payload["violation"])
        assert rechecked.status == "incompatible"
        assert rechecked.violation == result.violation
        with pytest.raises(ValueError, match="sides differ"):
            verify_compatibility_violation(graph, behavior, result.method, replace(result.violation, rhs=0))
        with pytest.raises(ValueError, match="direction"):
            verify_compatibility_violation(graph, behavior, result.method, replace(result.violation, required_relation="greater"))
        with pytest.raises(ValueError):
            verify_compatibility_violation(graph, behavior, result.method, replace(result.violation, group_outputs=((8,),)*len(result.violation.groups)))
        uniform = RationalBehavior(behavior.scenario, (Fraction(1, 8),)*8)
        with pytest.raises(ValueError):
            verify_compatibility_violation(graph, uniform, result.method, result.violation)
        payload["violation"]["lhs"] = 0.25
        with pytest.raises(ValueError):
            verify_compatibility_violation(graph, behavior, result.method, payload["violation"])
    document = behavior_to_dict(behavior)
    document["probabilities"][0] = 0.5
    with pytest.raises(ValueError):
        behavior_from_dict(document)
