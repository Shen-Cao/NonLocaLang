"""Exact independent oracle from the loop sectors in arXiv:2606.12363v1, App. E.

This calculation never permutes mode masks, constructs a density, or calls
the evaluator's measurement routine. Factor disjoint source cycles. In a cycle
with uneven token counts the split is uniquely determined, if possible. In
the all-one sector the all-kept and all-sent paths interfere as a GHZ state.
"""
from fractions import Fraction
from functools import cache
from itertools import product
from math import prod

import pytest

from nonlocalgames.protocols.fermion_reference import reference_arena, reference_target
from nonlocalgames.protocols.five_route_score import five_route_score
from nonlocalgames.protocols.quadratic import Quadratic
from nonlocalgames.protocols.quadratic_behavior import QuadraticBehavior, QuadraticWiringTarget


# These lists follow the physical directed arrows, not a permutation API.
LOOPS = {(2, 0, 1): ((0, 1, 2),), (1, 2, 0): ((0, 2, 1),),
         (1, 0, 2): ((0, 1), (2,)), (2, 1, 0): ((0, 2), (1,)),
         (0, 2, 1): ((1, 2), (0,))}


@cache
def _one_token(party, question, answer):
    # Basis |kept>, |incoming>; E=(I + label*(n_x X + n_z Z))/2.
    h = Quadratic(0, Fraction(1, 2))
    if party == 0:
        nx, nz = h, (-1)**question * h
    else:
        nx, nz = Quadratic(question), Quadratic(1-question)
    label = 1 if answer == 1 else -1
    return (1 + label*nz)/2, (1 - label*nz)/2, label*nx/2


@cache
def _loop_probability(cycle, questions, answers, graded, real_coherence=1):
    counts = tuple((0, 1, 1, 2)[a] for a in answers)
    one = {j: _one_token(p, x, a) for j, (p, x, a) in
           enumerate(zip(cycle, questions, answers)) if counts[j] == 1}
    total = Quadratic(0)
    for sent in product((0, 1), repeat=len(cycle)):
        if counts == tuple(sent[j-1] + (1-flag) for j, flag in enumerate(sent)):
            # Diagonal entry 0 = kept, 1 = incoming. Vacuum/double are certain.
            total += prod(entries[sent[j-1]] for j, entries in one.items())
    if all(n == 1 for n in counts):
        # Replacing one coherent path by the other cycles all particles once.
        relative_phase = (-1)**(len(cycle)-1) if graded else 1
        total += 2 * relative_phase * real_coherence * prod(entries[2] for entries in one.values())
    return total / 2**len(cycle)


@pytest.mark.parametrize("statistics,graded", [("fermionic", True), ("ordinary_tensor", False),
                                              ("fermionic_even_pair", False)])
def test_all_2560_cells_match_exact_analytical_cycle_oracle(statistics, graded):
    target = reference_target(statistics=statistics)
    checked = 0
    for route, branch in zip(target.arena.routes, target.behaviors):
        events = product(branch.scenario.inputs, branch.scenario.outputs)
        for (x, a), actual in zip(events, branch.probabilities, strict=True):
            expected = prod(_loop_probability(c, tuple(x[i] for i in c), tuple(a[i] for i in c),
                                              graded) for c in LOOPS[route])
            assert expected == actual, (route, x, a)
            checked += 1
    assert checked == 2560


@pytest.mark.parametrize("graded", [False, True])
def test_full_phase_or_attenuation_tables_replay_their_scoped_scores(graded):
    if graded:
        # Independent single-mode sources: attenuate only each off-diagonal by q.
        q = Fraction(1, 2)
        by_length = {n: q**n for n in (1, 2, 3)}
        expected = (2*q**3 + 3*q**2)/5
    else:
        # A balanced ordinary source phase with cos(phi)=-1/3. The complete
        # tables stay in Q(sqrt(2)) because all detector matrices are real.
        c = Fraction(-1, 3)
        by_length = {1: c, 2: 2*c*c-1, 3: 4*c**3-3*c}
        expected = Fraction(109, 135)  # exceeds 3/5, below (1+sqrt(5))/4
    arena = reference_arena()
    branches = {}
    for route, loops in LOOPS.items():
        events = product(arena.scenario.inputs, arena.scenario.outputs)
        values = (prod(_loop_probability(cycle, tuple(x[i] for i in cycle),
                                         tuple(a[i] for i in cycle), graded, by_length[len(cycle)])
                       for cycle in loops) for x, a in events)
        branches[route] = QuadraticBehavior(arena.scenario, values)
    result = five_route_score(QuadraticWiringTarget(arena, branches))
    assert result.score == expected
    controls = result.to_dict()["comparisons"]
    assert controls["fixed_local_sign_swaps"]["exceeds_upper_bound"] == (not graded)
    assert not controls["one_token_equatorial_phases"]["exceeds_upper_bound"]
