"""Two independent checks: contraction vs Eq. (11), and an exact classical control."""
from fractions import Fraction
from itertools import permutations, product
from math import lcm, prod

import pytest

from nonlocalgames import Scenario
from nonlocalgames.networks import (
    RGB4_NETWORK, RGB4_SCENARIO, ClassicalNetworkStrategy, RationalBehavior, SourceNetwork,
    behavior_from_dict, behavior_to_dict, rgb4_behavior, rgb4_dephased_strategy, rgb4_token_counting_margin,
    triangle_finner,
)


def _eq11(*, coherent):
    """Eq. (11) by sectors, without source bits, port maps, or projector tables."""
    u, v = (Fraction(12, 13), Fraction(5, 13)), (Fraction(5, 13), Fraction(-12, 13))
    table = {}
    for indices in product(range(2), repeat=3):
        left = prod(u[i] for i in indices)
        right = prod(v[i] for i in indices)
        table[tuple(i+1 for i in indices)] = (
            (left + right)**2 if coherent else left**2 + right**2) / 8
    for i in range(2):
        for event, amplitude in (((i+1, 0, 3), u[i]), ((i+1, 3, 0), v[i])):
            for offset in range(3):
                table[event[offset:] + event[:offset]] = amplitude**2 / 8
    return tuple(table.get(event, Fraction(0)) for event in product(range(4), repeat=3))


def _move(behavior, src, dst, amount):
    values = list(behavior.probabilities)
    values[behavior.scenario.output_index(src)] -= amount
    values[behavior.scenario.output_index(dst)] += amount
    return RationalBehavior(behavior.scenario, values)


@pytest.mark.parametrize("coherent", [True, False], ids=["bell_sources", "dephased"])
def test_all_64_cells_and_coarse_law(coherent):
    assert RGB4_NETWORK.sources == ((1, 2), (2, 0), (0, 1))
    assert RGB4_SCENARIO == Scenario((1, 1, 1), (4, 4, 4))
    if coherent:
        behavior = rgb4_behavior()
        assert lcm(*(p.denominator for p in behavior.probabilities)) == 38_614_472
    else:
        model = rgb4_dephased_strategy()
        assert model.network == RGB4_NETWORK
        assert model.source_probabilities == ((Fraction(1, 2), Fraction(1, 2)),) * 3
        behavior = RationalBehavior(model.scenario, model.probabilities())
    expected = _eq11(coherent=coherent)
    assert behavior.probabilities == expected
    assert len(expected) == 64 and sum(p != 0 for p in expected) == 20
    assert behavior_from_dict(behavior_to_dict(behavior)) == behavior

    coarse = {}
    for event, p in zip(behavior.scenario.outputs, behavior.probabilities, strict=True):
        if p:
            counts = tuple((0, 1, 1, 2)[i] for i in event)
            coarse[counts] = coarse.get(counts, 0) + p
    assert coarse == {(1, 1, 1): Fraction(1, 4),
                      **{event: Fraction(1, 8) for event in permutations((0, 1, 2))}}
    assert triangle_finner(RGB4_NETWORK, behavior).status == "not_ruled_out"
    expected_margin = Fraction(115_125 if coherent else -316_875, 38_614_472)
    assert rgb4_token_counting_margin(RGB4_NETWORK, behavior) == expected_margin


def test_classical_bound_can_be_saturated_exactly():
    # Each party directly announces its local occupation bits (00,01,10,11).
    # The framework gives Bob his two source variables in alpha,gamma order.
    model = ClassicalNetworkStrategy.from_response_rules(
        RGB4_NETWORK, RGB4_SCENARIO, ((Fraction(1, 2), Fraction(1, 2)),) * 3,
        (lambda x, bg, a: a == 2*(1-bg[0]) + bg[1],
         lambda x, ag, b: b == 2*(1-ag[1]) + ag[0],
         lambda x, ab, c: c == 2*(1-ab[0]) + ab[1]))
    behavior = RationalBehavior(RGB4_SCENARIO, model.probabilities())
    assert rgb4_token_counting_margin(RGB4_NETWORK, behavior) == 0


@pytest.mark.parametrize(("src", "dst", "coefficient"), [
    ((1, 3, 0), (2, 3, 0), 1), ((0, 1, 3), (0, 2, 3), 1), ((3, 0, 1), (3, 0, 2), 1),
    ((1, 0, 3), (2, 0, 3), 0), ((2, 2, 2), (1, 2, 2), 1),
])
def test_asymmetric_refinements_do_not_assume_cyclic_symmetry(src, dst, coefficient):
    table = rgb4_behavior()
    epsilon = Fraction(1, 1_000_000)
    changed = _move(table, src, dst, epsilon)
    assert rgb4_token_counting_margin(RGB4_NETWORK, changed) == (
        Fraction(115_125, 38_614_472) + coefficient * epsilon)


def test_refuse_wrong_scope_and_even_tiny_coarse_drift():
    behavior = rgb4_behavior()
    with pytest.raises(ValueError, match="independent source"):
        rgb4_token_counting_margin(SourceNetwork(3, ((0, 1, 2),)), behavior)
    with pytest.raises(TypeError, match="RationalBehavior"):
        rgb4_token_counting_margin(RGB4_NETWORK, behavior.probabilities)
    one_outcome = RationalBehavior(Scenario((1, 1, 1), (1, 1, 1)), (1,))
    with pytest.raises(ValueError, match="four outcomes"):
        rgb4_token_counting_margin(RGB4_NETWORK, one_outcome)
    # Without the premise an all-zero-output classical strategy would appear to violate.
    all_zero = RationalBehavior(RGB4_SCENARIO, (1,) + (0,) * 63)
    with pytest.raises(ValueError, match="exact fair-token coarse law"):
        rgb4_token_counting_margin(RGB4_NETWORK, all_zero)
    # Same support and conserved total token count, wrong probabilities.
    changed = _move(behavior, (1, 3, 0), (3, 0, 1), Fraction(1, 2**100))
    with pytest.raises(ValueError, match="exact fair-token coarse law"):
        rgb4_token_counting_margin(RGB4_NETWORK, changed)
