"""Six-mode trusted rewiring strategy of Moradi Kalarde et al., arXiv:2606.12363v1."""
from fractions import Fraction
from itertools import permutations, product

from ..core.game import Scenario
from .quadratic import Quadratic
from .quadratic_behavior import QuadraticBehavior, QuadraticWiringTarget
from .wiring import WiringArena


def reference_arena():
    """Five hidden permutations, with output labels (0, 1+, 1−, 2) in that order."""
    scenario = Scenario((2, 2, 2), (4, 4, 4))
    routes = (route for route in permutations(range(3)) if route != (0, 1, 2))
    return WiringArena(scenario, routes)


def _density_terms(route, fermionic):
    """Nonzero contributions to the density, scaled by eight, in lab mode order."""
    columns = []
    for sent in product((False, True), repeat=3):
        # Source creation operators appear in A,B,C order. Final modes are
        # (received, kept) for each lab, in A,B,C order (B.2 and B.6).
        occupied = tuple(2*route.index(i) if sent[i] else 2*i+1 for i in range(3))
        inversions = sum(k > later for j, k in enumerate(occupied) for later in occupied[j+1:])
        sign = -1 if fermionic and inversions % 2 else 1
        local = tuple(2*(2*i in occupied) + (2*i+1 in occupied) for i in range(3))
        counts = tuple(n.bit_count() for n in local)
        columns.append((sign, local, counts))
    # Each local effect preserves particle number; only pairs in the same
    # local number sectors can contribute to the Born probability.
    return tuple((s*t, left, right) for s, left, counts in columns
                 for t, right, other_counts in columns if counts == other_counts)


def _effect_entry(party, question, answer, row, column):
    """Return the (rational, radical) integer numerators of four times one entry."""
    if answer in (0, 3):
        return 4 * int(row == column == answer), 0
    if row not in (1, 2) or column not in (1, 2):
        return 0, 0
    # One-particle basis (kept-only |01>, received-only |10>). The measured
    # observables are (X ± Z)/sqrt(2) for Alice, and Z or X for Bob and Charlie.
    unit = int(row == column)
    z = unit if row == 1 else -unit
    x = int(row != column)
    sign = 1 if answer == 1 else -1
    if party == 0:
        return 2*unit, sign*(x + (z if question == 0 else -z))
    return 2*(unit + sign*(z if question == 0 else x)), 0


def _effects():
    # Index each local 4-by-4 effect by 4*row+column in occupation order.
    return {(i, x, a): tuple(_effect_entry(i, x, a, r, c) for r, c in product(range(4), repeat=2))
            for i, x, a in product(range(3), range(2), range(4))}


def _behavior(scenario, route, effects, fermionic):
    terms = [(sign, *(4*r+c for r, c in zip(left, right)))
             for sign, left, right in _density_terms(route, fermionic)]
    probabilities = []
    denominator = 8 * 4**3  # density factor times three local projector denominators
    outputs = scenario.outputs
    for x in scenario.inputs:
        for a in outputs:
            alice, bob, charlie = (effects[i, x[i], a[i]] for i in range(3))
            rational = radical = 0
            for sign, i, j, k in terms:
                # Only Alice's measurement has irrational entries.
                weight = sign * bob[j][0] * charlie[k][0]
                rational += weight * alice[i][0]
                radical += weight * alice[i][1]
            probabilities.append(Quadratic(Fraction(rational, denominator), Fraction(radical, denominator)))
    return QuadraticBehavior(scenario, probabilities)


def reference_target(*, statistics='fermionic'):
    """Evaluate the full unconditioned family with the B.4 preparations and B.17–18 readout.

    'fermionic' applies the graded permutation to the sent modes. 'ordinary_tensor'
    performs the same experiment with ordinary permutation and zero preparation
    phases. Each setting supplies a particular strategy's exact statistics.
    """
    if statistics not in ('fermionic', 'ordinary_tensor'):
        raise ValueError('statistics must be fermionic or ordinary_tensor')
    arena = reference_arena()
    effects = _effects()
    branches = {route: _behavior(arena.scenario, route, effects, statistics == 'fermionic')
                for route in arena.routes}
    return QuadraticWiringTarget(arena, branches)
