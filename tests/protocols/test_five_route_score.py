from fractions import Fraction as F
from math import isqrt

import pytest

from nonlocalgames import Scenario
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.protocols.fermion_reference import (
    ReadoutContext, balanced_strategy_target, reference_arena, reference_target,
)
from nonlocalgames.protocols.five_route_score import five_route_score
from nonlocalgames.protocols.quadratic import Quadratic as Q
from nonlocalgames.protocols.quadratic_behavior import QuadraticBehavior, QuadraticWiringTarget
from nonlocalgames.protocols.quadratic_wire import target_id
from nonlocalgames.protocols.wiring import WiringArena


def _deterministic(arena, rule):
    branches = {}
    for route in arena.routes:
        probabilities = []
        for x in arena.scenario.inputs:
            output = rule(route, x)
            probabilities.extend(Q(int(a == output)) for a in arena.scenario.outputs)
        branches[route] = QuadraticBehavior(arena.scenario, probabilities)
    return QuadraticWiringTarget(arena, branches)


def _contrasts(result):
    return (result.abc, result.acb, result.ab, result.ac, result.bc)


def _exceeds(result):
    comparisons = result.to_dict()["comparisons"]
    return (comparisons["fixed_local_sign_swaps"]["exceeds_upper_bound"],
            comparisons["one_token_equatorial_phases"]["exceeds_upper_bound"])


@pytest.mark.parametrize("statistics, expected", [
    ("fermionic", (1, 1, -1, -1, -1)), ("ordinary_tensor", (1,)*5),
    ("fermionic_even_pair", (1,)*5),
])
def test_raw_reference_contrasts_and_scoped_comparisons(statistics, expected):
    t = reference_target(statistics=statistics)
    r = five_route_score(t)
    assert r.target_id == target_id(t)
    assert _contrasts(r) == expected
    single_mode = statistics == "fermionic"
    assert r.score == (1 if single_mode else F(-1, 5))
    assert _exceeds(r) == (single_mode, single_mode)


@pytest.mark.parametrize("statistics", ["ordinary_tensor", "fermionic_even_pair"])
def test_allowed_calibration_attains_fixed_swap_cap_not_a_strict_violation(statistics):
    swaps = [ReadoutContext(receiver=i, incoming_sender=s, question=x)
             for i, s, x in ((0, 2, 0), (0, 2, 1), (1, 0, 1))]
    result = five_route_score(balanced_strategy_target(statistics=statistics, swap_sign_labels=swaps))
    assert _contrasts(result) == (1, 1, -1, -1, 1)  # only BC misses
    assert result.score == F(3, 5) and _exceeds(result) == (False, False)
    # Opposite swaps for Alice's two questions erase BOTH contrasts using that sender.
    unpaired = five_route_score(balanced_strategy_target(statistics=statistics,
                                                        swap_sign_labels=swaps[:1]))
    assert _contrasts(unpaired) == (0, 1, 1, 0, 1)


def test_ignores_both_neutral_labels_and_sums_over_all_outputs_at_idle_sites():
    arena = reference_arena()
    neutral = _deterministic(arena, lambda r, x: (3, 0, 3))
    assert _contrasts(five_route_score(neutral)) == (0,)*5  # no conditional 0/0
    # In AB both active readouts are + exactly at B's question 1. In BC, A may
    # report vacuum. In both cases an idle lab must not be filtered to 1+/1-.
    def rule(route, x):
        if route == (1, 0, 2):
            return (1, x[1], 3)
        if route == (0, 2, 1):
            return (0, x[1], 2*x[2])
        return (0, 0, 0)

    r = five_route_score(_deterministic(arena, rule))
    assert _contrasts(r) == (0, 0, Q(0, 2), 0, -2)
    assert r.score == Q(F(2, 5), F(-2, 5))


def _near_phase_cap(step):
    # Rational neighbours of the positive irrational root that binary64 cannot separate.
    d = 2**96
    return Q(F(d + isqrt(5*d*d) + step, 4*d))


@pytest.mark.parametrize("strength, flags", [(Q(0, F(1, 2)), (True, False)),
                                             (Q(0, F(3, 5)), (True, True)),
                                             (_near_phase_cap(0), (True, False)),
                                             (_near_phase_cap(1), (True, True))])
def test_unconditioned_score_dilutes_linearly_in_the_radical_field(strength, flags):
    t = reference_target()
    # Branchwise mixture with all-vacuum outputs: conditional correlations on a
    # sign sector would not change. A fixed raw score must scale with its mass.
    branches = {}
    for route, b in zip(t.arena.routes, t.behaviors, strict=True):
        width = len(b.scenario.outputs)
        values = (strength*p + (1-strength if i % width == 0 else 0)
                  for i, p in enumerate(b.probabilities))
        branches[route] = QuadraticBehavior(b.scenario, values)
    result = five_route_score(QuadraticWiringTarget(t.arena, branches))
    assert result.score == strength and _exceeds(result) == flags


def test_signed_score_is_not_an_acceptance_probability_or_a_general_classical_bound():
    arena = reference_arena()
    # Constant local sign outputs, with the SAME allowed directed swaps as above.
    # This is a classical strategy, but not either fixed-sector measurement family.
    def classical(route, x):
        return (2 if route[0] == 2 else 1, 2 if (route[1], x[1]) == (0, 1) else 1, 1)

    r = five_route_score(_deterministic(arena, classical))
    assert r.score == Q(F(-2, 5), F(12, 5)) and _exceeds(r) == (True, True)
    # A negative number below the negative root of 4s²-2s-1 also makes that
    # polynomial positive, but is NOT above the positive phase-family cap.
    negative = five_route_score(_deterministic(arena, lambda r, x: (2, 2, 2)))
    assert negative.score == Q(F(-2, 5), F(-12, 5)) and _exceeds(negative) == (False, False)


def test_requires_entire_reference_arena_and_bounds_coefficients():
    arena = reference_arena()
    def const(route, questions):
        return (1, 1, 1)

    with pytest.raises(TypeError, match="quadratic"):
        five_route_score(_deterministic(arena, const).as_rational())
    for bad in (WiringArena(arena.scenario, ((0, 1, 2), *arena.routes[1:])),
                WiringArena(Scenario((1, 2, 2), (4, 4, 4)), arena.routes),
                WiringArena(Scenario((2, 2, 2), (4, 4, 3)), arena.routes)):
        with pytest.raises(ValueError, match="reference five routes"):
            five_route_score(_deterministic(bad, const))
    target = _deterministic(arena, const)
    with pytest.raises(ScalarBudgetExceeded, match="max_scalar_bits"):
        five_route_score(target, max_scalar_bits=2)
    with pytest.raises(ValueError, match="positive integer"):
        five_route_score(target, max_scalar_bits=True)
