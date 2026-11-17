"""Exact checks of the trusted distributions in arXiv:2606.12363v1, B.9–12."""
from collections import defaultdict
from fractions import Fraction
from itertools import permutations

import pytest

from nonlocalgames.protocols.examples import ABC, BC
from nonlocalgames.protocols.fermion_reference import reference_arena, reference_target
from nonlocalgames.protocols.quadratic import Quadratic


ACB = (1, 2, 0)


@pytest.fixture(scope='module')
def families():
    return reference_target(), reference_target(statistics='ordinary_tensor')


def _branch(target, route):
    return target.behaviors[target.arena.routes.index(route)]


def test_five_hidden_permutations_and_persistent_local_ports():
    arena = reference_arena()
    assert arena.routes == tuple(r for r in permutations(range(3)) if r != (0, 1, 2))
    assert arena.scenario.entries * len(arena.routes) == 2560
    for route in arena.routes:
        for i in range(3):
            assert arena.held_ports(route, i) == ((i, 'kept'), (route[i], 'sent'))
            assert arena.local_view(route, i, 1) == (i, 1, route[i])


def test_unconditioned_token_counts_for_every_question_and_wiring(families):
    weights = (0, 1, 1, 2)
    for target in families:
        for route, branch in zip(target.arena.routes, target.behaviors):
            fixed = [i for i in range(3) if route[i] == i]
            expected = {(1, 1, 1): Fraction(1, 2 if fixed else 4)}
            expected.update({n: Fraction(1, 4 if fixed else 8) for n in permutations(range(3))
                             if not fixed or n[fixed[0]] == 1})
            for i, _ in enumerate(branch.scenario.inputs):
                counts = defaultdict(lambda: Quadratic(0))
                for a, p in zip(branch.scenario.outputs, branch.probabilities[64*i:64*(i+1)]):
                    if p != 0:
                        counts[tuple(weights[label] for label in a)] += p
                assert counts == expected


def test_odd_loops_agree_and_even_loops_have_observable_relative_sign(families):
    fermion, tensor = families
    for route in fermion.arena.routes:
        same = _branch(fermion, route) == _branch(tensor, route)
        assert same == all(i != sender for i, sender in enumerate(route))
    plus_sign = Quadratic(Fraction(1, 32), Fraction(1, 64))
    minus_sign = Quadratic(Fraction(1, 32), Fraction(-1, 64))
    for route in (ABC, ACB):
        outcomes = _branch(fermion, route).marginal((0, 1, 2), (0, 1, 1))
        assert outcomes[1, 1, 1] == plus_sign
        assert outcomes[1, 2, 1] == minus_sign
    for target, agreeing in ((fermion, False), (tensor, True)):
        outcomes = _branch(target, BC).marginal((1, 2), (0, 1, 1))
        for b in (1, 2):
            for c in (1, 2):
                assert outcomes[b, c] == (Fraction(1, 4) if (b == c) == agreeing else 0)


def test_isolated_party_recovers_its_own_coherent_one_particle_state(families):
    for target in families:
        for route, branch in zip(target.arena.routes, target.behaviors):
            for i in range(3):
                if route[i] != i:
                    continue
                for x in branch.scenario.inputs:
                    local = branch.marginal((i,), x)
                    plus = (Quadratic(Fraction(1, 2), Fraction(1, 4)) if i == 0
                            else Fraction(1, 2) if x[i] == 0 else 1)
                    assert local == {(0,): 0, (1,): plus, (2,): 1-plus, (3,): 0}


def test_one_particle_readout_retains_the_directed_port_convention(families):
    # On (A=1, B=2, C=0), A's token is received in ABC and retained in ACB.
    for target in families:
        for route, sign in ((ABC, -1), (ACB, 1)):
            branch = _branch(target, route)
            for x in branch.scenario.inputs:
                p = branch.marginal((0, 1, 2), x)[1, 3, 0]
                assert p == Quadratic(Fraction(1, 16), Fraction(sign*(-1)**x[0], 32))


def test_only_named_carrier_models_can_select_reference_statistics():
    with pytest.raises(ValueError, match='statistics'):
        reference_target(statistics='quantum')
