"""Exact fixed-source strategies for the five hidden wirings of arXiv:2606.12363v1."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product

from ..core.game import Scenario
from .graded import relay_mode_permutation
from .quadratic import Quadratic
from .quadratic_behavior import QuadraticBehavior, QuadraticWiringTarget
from .wiring import WiringArena, _index


REFERENCE_STATISTICS = ("fermionic", "ordinary_tensor", "fermionic_even_pair")


@dataclass(frozen=True, kw_only=True, slots=True)
class ReadoutContext:
    """All and only the classical labels visible at a measurement in this experiment."""

    receiver: int
    incoming_sender: int
    question: int

    def __post_init__(self):
        for name, size in (("receiver", 3), ("incoming_sender", 3), ("question", 2)):
            object.__setattr__(self, name, _index(getattr(self, name), size, name))


def reference_arena():
    """Five hidden permutations, with output labels (0, 1+, 1−, 2) in that order."""
    scenario = Scenario((2, 2, 2), (4, 4, 4))
    routes = (route for route in permutations(range(3)) if route != (0, 1, 2))
    return WiringArena(scenario, routes)


def _source_masks(width):
    """The eight equal-amplitude terms of three independent balanced preparations."""
    full = (1 << width) - 1
    return tuple(sum(full << ((2*i + role)*width) for i, role in enumerate(roles))
                 for roles in product((0, 1), repeat=3))  # 0 = sent, 1 = kept


def _density_terms(route, source_masks, width, graded):
    """Route physical modes, then restrict the readout to the reached code subspace."""
    permutation = relay_mode_permutation(route, modes_per_port=width)
    full = (1 << width) - 1
    logical = {0: 0, full: 1}  # empty/full packet, not its physical particle count
    columns = []
    for mask in source_masks:
        occupied, sign = permutation.apply(mask)
        packets = tuple(logical[(occupied >> (j*width)) & full] for j in range(6))
        local = tuple(2*incoming + kept for incoming, kept in zip(packets[::2], packets[1::2]))
        counts = tuple(n.bit_count() for n in local)
        columns.append((sign if graded else 1, local, counts))
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


def _effects(arena, swaps):
    # Each reachable context gets exactly one readout, reused in every route.
    projectors = {(i, x): tuple(tuple(_effect_entry(i, x, a, r, c)
                                      for r, c in product(range(4), repeat=2)) for a in range(4))
                  for i, x in product(range(3), range(2))}
    views = {ReadoutContext(receiver=i, incoming_sender=s, question=x)
             for route in arena.routes for i, s in enumerate(route) for x in range(2)}
    by_view = {}
    for v in views:
        zero, plus, minus, two = projectors[v.receiver, v.question]
        by_view[v] = (zero, minus, plus, two) if v in swaps else (zero, plus, minus, two)
    return by_view


def _behavior(scenario, route, effects, source_masks, width, graded):
    terms = [(sign, *(4*r+c for r, c in zip(left, right)))
             for sign, left, right in _density_terms(route, source_masks, width, graded)]
    readouts = tuple(tuple(effects[ReadoutContext(receiver=i, incoming_sender=s, question=x)]
                           for x in range(2)) for i, s in enumerate(route))
    probabilities = []
    denominator = 8 * 4**3  # density factor times three local projector denominators
    outputs = scenario.outputs
    for x in scenario.inputs:
        local = tuple(readouts[i][x[i]] for i in range(3))
        for a in outputs:
            alice, bob, charlie = (local[i][a[i]] for i in range(3))
            rational = radical = 0
            for sign, i, j, k in terms:
                # Only Alice's measurement has irrational entries.
                weight = sign * bob[j][0] * charlie[k][0]
                rational += weight * alice[i][0]
                radical += weight * alice[i][1]
            probabilities.append(Quadratic(Fraction(rational, denominator), Fraction(radical, denominator)))
    return QuadraticBehavior(scenario, probabilities)


def balanced_strategy_target(*, statistics="fermionic", swap_sign_labels=()):
    """Evaluate a fixed balanced source at each origin and a restricted local readout.

    Optional ReadoutContext keys swap the two one-token output labels only.
    Sources never depend on the route. No route tag, outgoing destination,
    remote question or independently adjustable branch measurement is accepted.
    """
    if statistics not in REFERENCE_STATISTICS:
        raise ValueError(f"statistics must be one of {REFERENCE_STATISTICS}")
    swaps = set()
    for view in swap_sign_labels:
        # A subclass can add an outgoing destination or a route tag to its
        # key. Reject it rather than accepting a swap that cannot be looked up.
        if type(view) is not ReadoutContext:
            raise TypeError("Each sign-label swap needs a local ReadoutContext")
        if view in swaps:
            raise ValueError("Duplicate readout context")
        swaps.add(view)
    arena = reference_arena()
    width = 2 if statistics == "fermionic_even_pair" else 1
    source_masks = _source_masks(width)
    effects = _effects(arena, swaps)
    branches = {route: _behavior(arena.scenario, route, effects, source_masks, width,
                                 statistics != "ordinary_tensor")
                for route in arena.routes}
    return QuadraticWiringTarget(arena, branches)


def reference_target(*, statistics="fermionic"):
    """The full unconditioned family for one named strategy, with no relabeling.

    'fermionic' is the six-mode reference. 'ordinary_tensor' uses an unsigned
    permutation and zero preparation phases. 'fermionic_even_pair' encodes
    each port as |00>, |11> and applies the graded permutation on twelve modes.
    """
    return balanced_strategy_target(statistics=statistics)
