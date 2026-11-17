"""Independent single-particle optics check of the six-mode reference tables."""
from fractions import Fraction
from itertools import permutations, product
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from nonlocalgames.protocols.fermion_reference import reference_target


@pytest.fixture(scope='module')
def fermionic():
    return reference_target(statistics='fermionic')


def _slater_row(route, questions):
    # Columns are the three occupied source orbitals, in A, B, C order.
    # Receiver i holds mode R from its sender and its own retained mode L.
    orbitals = np.zeros((6, 3))
    for i, sender in enumerate(route):
        orbitals[2*i, sender] = 1/sqrt(2)
        orbitals[2*i+1, i] = 1/sqrt(2)
    angles = ((3*pi/8, pi/8), (pi/2, pi/4), (pi/2, pi/4))
    for party, question in enumerate(questions):
        angle = angles[party][question]
        rotation = np.array([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]])
        site = slice(2*party, 2*party+2)
        orbitals[site] = rotation @ orbitals[site]
    # The rotated R detector sees cos(alpha)|R>+sin(alpha)|L>, the B.6 plus ray.
    occupied = ((), (0,), (1,), (0, 1))
    row = []
    for answers in product(range(4), repeat=3):
        modes = [2*i+j for i, answer in enumerate(answers) for j in occupied[answer]]
        probability = abs(np.linalg.det(orbitals[modes, :]))**2 if len(modes) == 3 else 0.0
        row.append(probability)
    return row


def test_every_fermionic_probability_matches_the_three_orbital_determinant(fermionic):
    routes = tuple(route for route in permutations(range(3)) if route != (0, 1, 2))
    assert fermionic.arena.routes == routes
    assert len(fermionic.behaviors) == 5
    for route, behavior in zip(routes, fermionic.behaviors):
        expected = [_slater_row(route, x) for x in product(range(2), repeat=3)]
        actual = [float(p.a) + sqrt(2)*float(p.b) for p in behavior.probabilities]
        np.testing.assert_allclose(np.reshape(actual, (8, 64)), expected, rtol=0, atol=1e-14,
                                   err_msg=f'route {route}')


def test_rotated_plus_detector_follows_alice_input_and_cycle_orientation(fermionic):
    # The coarse event (1,0,2) singles out one of the eight source assignments.
    # Reversing the cycle exchanges Alice's occupied port and hence u² and v².
    scenario = fermionic.arena.scenario
    event = scenario.output_index((1, 0, 3))  # 1+, zero, two
    for route, sign in (((2, 0, 1), 1), ((1, 2, 0), -1)):
        behavior = fermionic.behaviors[fermionic.arena.routes.index(route)]
        for x_a in range(2):
            index = 64*scenario.input_index((x_a, 0, 0)) + event
            value = behavior.probabilities[index]
            assert (value.a, value.b) == (Fraction(1, 16), Fraction(sign*(-1)**x_a, 32))


def test_specified_ordinary_and_fermionic_witnesses_differ_on_two_cycle(fermionic):
    ordinary = reference_target(statistics='ordinary_tensor')
    route = (1, 0, 2)
    for target, sign in ((ordinary, 1), (fermionic, -1)):
        branch = target.behaviors[target.arena.routes.index(route)]
        # Summing over C includes every outcome of the laboratory receiving its own port.
        value = branch.marginal((0, 1), (0, 1, 0))[1, 1]
        assert (value.a, value.b) == (Fraction(1, 8), Fraction(sign, 16))
