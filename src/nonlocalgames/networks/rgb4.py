"""Exact static RGB4 triangle at cos(theta)=12/13, sin(theta)=5/13.

Output indices 0, 1, 2, 3 denote 0, 1_0, 1_1, 2. See Eq. (11) of
Sekatski, Boreiri and Brunner, https://arxiv.org/abs/2209.09921v2.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import permutations, product
from math import prod

from ..core.game import Scenario
from .behavior import RationalBehavior
from .classical import ClassicalNetworkStrategy
from .compatibility import validate_triangle
from .structure import SourceNetwork

RGB4_NETWORK = SourceNetwork(3, ((1, 2), (2, 0), (0, 1)))  # alpha=BC, beta=CA, gamma=AB
RGB4_SCENARIO = Scenario((1, 1, 1), (4, 4, 4))
# Local ket factors: A=(beta,gamma), B=(gamma,alpha), C=(alpha,beta).
_LOCAL_SOURCES = ((1, 2), (2, 0), (0, 1))
# Each row is a measurement ket times 13, in local basis 00,01,10,11.
_KETS = ((13, 0, 0, 0), (0, 12, 5, 0), (0, 5, -12, 0), (0, 0, 0, 13))


def _basis_index(left, right):
    # Each source bit is the occupation at its FIRST endpoint in RGB4_NETWORK.
    # Each party is the second endpoint on its left edge, first on its right.
    return 2 * (1 - left) + right


def rgb4_behavior() -> RationalBehavior:
    """Contract three independent (|01>+|10>)/sqrt(2) sources exactly.

    Sum the eight product-state amplitudes *before* squaring. Integer ket
    numerators leave one common Born denominator; no dense state or rounding.
    """
    terms = tuple(tuple(_basis_index(bits[left], bits[right]) for left, right in _LOCAL_SOURCES)
                  for bits in product(range(2), repeat=3))
    probabilities = []
    for outputs in RGB4_SCENARIO.outputs:
        amplitude = sum(prod(_KETS[a][b] for a, b in zip(outputs, basis)) for basis in terms)
        probabilities.append(Fraction(amplitude**2, 8 * 13**6))
    return RationalBehavior(RGB4_SCENARIO, probabilities)


def rgb4_dephased_strategy() -> ClassicalNetworkStrategy:
    """Realize the source-dephased control using three independent fair bits.

    A source bit selects |01> or |10>; each party samples the squared local
    ket entries for its incoming basis state. No shared selector or route.
    """
    def response_rule(party):
        incident = RGB4_NETWORK.incident_sources(party)  # framework table order
        left, right = (incident.index(source) for source in _LOCAL_SOURCES[party])

        def response(_setting, bits, outcome):
            basis = _basis_index(bits[left], bits[right])
            return Fraction(_KETS[outcome][basis]**2, 13**2)

        return response

    fair = (Fraction(1, 2), Fraction(1, 2))
    return ClassicalNetworkStrategy.from_response_rules(
        RGB4_NETWORK, RGB4_SCENARIO, (fair,) * 3, tuple(map(response_rule, range(3))))


def rgb4_token_counting_margin(network: SourceNetwork, behavior: RationalBehavior) -> Fraction:
    """Return 1/8 minus P(1_1,1_1,1_1) and the cyclic sum of P(1_0,2,0).

    Requires the exact fair-token coarse law. A positive margin excludes
    independent *classical* triangle sources without communication or global
    shared randomness, using Renou & Beigi's token-counting rigidity theorem
    (https://arxiv.org/abs/2202.00905, Thm. 1). Nonpositive is inconclusive.
    Not a noise-robust or unconditional inequality. See docs/rgb4.md.
    """
    validate_triangle(network)
    if not isinstance(behavior, RationalBehavior):
        raise TypeError("RGB4 comparison requires a RationalBehavior")
    if behavior.scenario != RGB4_SCENARIO:
        raise ValueError("RGB4 comparison requires three parties, no inputs and four outcomes each")

    counts = (0, 1, 1, 2)
    coarse = defaultdict(Fraction)
    for event, p in zip(behavior.scenario.outputs, behavior.probabilities, strict=True):
        coarse[tuple(counts[a] for a in event)] += p
    expected = {event: Fraction(1, 8) for event in permutations((0, 1, 2))}
    expected[(1, 1, 1)] = Fraction(1, 4)
    if any(p != expected.get(event, 0) for event, p in coarse.items()):
        raise ValueError("RGB4 comparison requires the exact fair-token coarse law")

    events = ((2, 2, 2), (1, 3, 0), (0, 1, 3), (3, 0, 1))
    return Fraction(1, 8) - sum(behavior.probabilities[RGB4_SCENARIO.output_index(event)]
                                for event in events)
