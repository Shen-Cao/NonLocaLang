from fractions import Fraction

import pytest

from nonlocalgames import Game, Scenario
from nonlocalgames.networks import ClassicalNetworkStrategy, SourceNetwork, triangle_xor_strategy


def test_source_order_and_invalid_networks():
    network = SourceNetwork(3, ((2, 0), (1, 2)))
    assert network.sources == ((2, 0), (1, 2))
    assert network.party_ports == (((0, 1),), ((1, 0),), ((0, 0), (1, 1)))
    assert network.incident_sources(2) == (0, 1)
    for parties, sources in [(0, ()), (True, ()), (2, ((),)), (2, ((0, 0),)), (2, ((2,),)), (2, ((True,),))]:
        with pytest.raises(ValueError):
            SourceNetwork(parties, sources)
    with pytest.raises(ValueError):
        network.incident_sources(-1)


def test_triangle_independent_bits_exact_behavior():
    strategy = triangle_xor_strategy()
    expected = tuple(Fraction(1, 4) if sum(a) % 2 == 0 else Fraction(0)
                     for a in strategy.scenario.outputs)
    assert strategy.probabilities() == expected
    parity = Game.from_rule(strategy.scenario, lambda x, a: sum(a) % 2 == 0)
    assert strategy.score(parity) == 1
    with pytest.raises(ValueError):
        strategy.probabilities(max_evaluations=1)


def test_private_randomness_without_sources():
    scenario = Scenario((1, 1), (2, 2))
    strategy = ClassicalNetworkStrategy(SourceNetwork(2, ()), scenario, (),
                                         ((Fraction(1, 3), Fraction(2, 3)), (Fraction(1, 4), Fraction(3, 4))))
    assert strategy.probabilities() == (Fraction(1, 12), Fraction(1, 4), Fraction(1, 6), Fraction(1, 2))


def test_unequal_latent_cardinalities_and_response_order():
    network = SourceNetwork(2, ((0, 1), (0,)))
    scenario = Scenario((2, 1), (3, 2))
    strategy = ClassicalNetworkStrategy.from_response_rules(
        network, scenario, ((Fraction(1, 4), Fraction(3, 4)), (Fraction(1, 3),)*3),
        (lambda x, values, a: a == (values[0] + values[1] + x) % 3,
         lambda x, values, a: a == values[0]))
    for x in scenario.inputs:
        for a in scenario.outputs:
            p = strategy.probabilities()[scenario.input_index(x)*6 + scenario.output_index(a)]
            assert p == (Fraction(1, 12) if a[1] == 0 else Fraction(1, 4))


def test_reject_correlated_source_payload_and_bad_responses():
    graph, scenario = SourceNetwork(3, ((0, 1), (1, 2))), Scenario((1, 1, 1), (1, 1, 1))
    for sources in [((Fraction(1, 2), 0, 0, Fraction(1, 2)),), ((1,), (0,)), ((1.0,), (1,))]:
        with pytest.raises((ValueError, TypeError)):
            ClassicalNetworkStrategy(graph, scenario, sources, ((1,),)*3)
    with pytest.raises(ValueError):
        ClassicalNetworkStrategy(graph, scenario, ((1,), (1,)), ((1,), (0,), (1,)))
    with pytest.raises(ValueError):
        ClassicalNetworkStrategy.from_response_rules(graph, scenario, ((1,), (1,)),
                                                     (lambda x, v, a: 1,)*3, max_entries=2)


def test_seeded_stochastic_bilocal_models_satisfy_nonlinear_inequality():
    import random
    from nonlocalgames.networks import BILOCAL_SCENARIO, bilocal_witness
    rng = random.Random(42)

    def distribution(size):
        weights = [rng.randint(1, 8) for _ in range(size)]
        return tuple(Fraction(w, sum(weights)) for w in weights)

    graph = SourceNetwork(3, ((0, 1), (1, 2)))
    for _ in range(10):
        sources = (distribution(2), distribution(3))
        responses = []
        for party, inputs, outputs in zip(range(3), (2, 1, 2), (2, 4, 2)):
            latent_count = 1
            for s in graph.incident_sources(party):
                latent_count *= len(sources[s])
            responses.append(tuple(p for _ in range(inputs*latent_count) for p in distribution(outputs)))
        strategy = ClassicalNetworkStrategy(graph, BILOCAL_SCENARIO, sources, responses)
        witness = bilocal_witness(strategy.probabilities())
        assert not witness.violates_bound
        assert witness.guarantee == "exact_rational_witness"
