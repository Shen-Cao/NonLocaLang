"""Rational controls for the sender-visible hidden-wiring experiment."""
from fractions import Fraction
from itertools import product

from ..core.game import Scenario
from ..networks.behavior import RationalBehavior
from .classical import ClassicalWiringStrategy
from .wiring import WiringArena, WiringTarget

# Each entry gives the sender at A, B and C respectively.
ABC = (2, 0, 1)
AC = (2, 1, 0)
BC = (0, 2, 1)


def inconsistent_alice_target():
    """Alice receives Charlie's port in both branches but her output laws disagree."""
    scenario = Scenario((1, 1, 1), (2, 1, 1))
    arena = WiringArena(scenario, (ABC, AC))
    return WiringTarget(arena, {ABC: RationalBehavior(scenario, (1, 0)),
                                AC: RationalBehavior(scenario, (0, 1))})


def bit_exchange_target():
    """Output parity is even on the participating cycle; fixed points output zero."""
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    arena = WiringArena(scenario, (ABC, AC, BC))
    supports = {ABC: ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
                AC: ((0, 0, 0), (1, 0, 1)), BC: ((0, 0, 0), (0, 1, 1))}
    behaviors = {}
    for route, support in supports.items():
        probabilities = tuple(Fraction(int(a in support), len(support)) for a in scenario.outputs)
        behaviors[route] = RationalBehavior(scenario, probabilities)
    return WiringTarget(arena, behaviors)


def bit_exchange_strategy():
    """Each party copies a private fair bit and compares its copy to the received bit."""
    arena = bit_exchange_target().arena
    joint_law = (Fraction(1, 2), 0, 0, Fraction(1, 2))
    response = tuple(int(output == (kept ^ incoming))
                     for kept, incoming, output in product(range(2), repeat=3))
    tables = {(receiver, sender): response for route in arena.routes
              for receiver, sender in enumerate(route)}
    return ClassicalWiringStrategy(arena, (2, 2, 2), (2, 2, 2), (joint_law,)*3, tables)
