"""Signed occupation permutations checked against elementary adjacent FSWAPs."""
from fractions import Fraction
from itertools import permutations, repeat

import pytest

from nonlocalgames.protocols.graded import ModePermutation, relay_mode_permutation
from nonlocalgames.protocols.wiring import WiringBudgetExceeded


def _adjacent_swaps(destinations):
    # Move labeled source wires one position at a time into destination order.
    wires = list(range(len(destinations)))
    swaps = []
    for target in range(len(wires)):
        source = destinations.index(target)
        at = wires.index(source)
        while at > target:
            wires[at-1], wires[at] = wires[at], wires[at-1]
            swaps.append(at-1)
            at -= 1
    return swaps


def _fswap(occupied, swaps):
    sign = 1
    for i in swaps:
        pair = (occupied >> i) & 3
        if pair == 3:
            sign = -sign
        elif pair in (1, 2):
            occupied ^= 3 << i
    return occupied, sign


def test_all_six_mode_permutations_and_input_masks_match_adjacent_swaps_and_invert():
    for destinations in permutations(range(6)):
        permutation = ModePermutation(destinations)
        inverse = permutation.inverse()
        swaps = _adjacent_swaps(destinations)
        for occupied in range(64):
            moved, sign = permutation.apply(occupied)
            assert (moved, sign) == _fswap(occupied, swaps)
            assert inverse.apply(moved) == (occupied, sign)


def test_three_cycle_directions_and_sequential_composition():
    p = ModePermutation((1, 2, 0))
    # Particles at 0,1 go to 1,2. The inverse sends them to 2,0, requiring a minus.
    assert p.apply(0b011) == (0b110, 1)
    assert p.inverse().apply(0b011) == (0b101, -1)
    q = ModePermutation((0, 2, 1))
    assert p.then(q) != q.then(p)
    for occupied in range(8):
        first, s1 = p.apply(occupied)
        second, s2 = q.apply(first)
        assert p.then(q).apply(occupied) == (second, s1*s2)
    # The hidden wire lists a sender per receiver, the opposite direction.
    assert relay_mode_permutation((1, 2, 0)).destinations == (4, 1, 0, 3, 2, 5)


def test_packet_swap_interferes_differently_on_odd_odd_and_even_empty_branches():
    swap = ModePermutation.from_packets((1, 0), (2, 2))
    # |10;10> + |11;00>, with equal positive amplitudes (bit zero = leftmost mode).
    odd, even = swap.apply(0b0101), swap.apply(0b0011)
    assert odd == (0b0101, -1)   # -|10;10>
    assert even == (0b1100, 1)   # +|00;11>
    # Project the normalized result onto (|10;10> + |00;11>)/sqrt(2).
    assert Fraction(odd[1] + even[1], 2)**2 == 0
    assert Fraction(abs(odd[1]) + abs(even[1]), 2)**2 == 1  # unsigned swap


def test_packet_expansion_keeps_internal_order_for_unequal_source_widths():
    assert ModePermutation.from_packets((2, 0, 1), (1, 3, 2)).destinations == (5, 0, 1, 2, 3, 4)
    empty = ModePermutation.from_packets((), ())
    assert empty.then(empty.inverse()).apply(0) == (0, 1)
    # A fat mode index and explicit bound should survive composition, without
    # any exponential state allocation or dependence on a machine-word mask.
    wide = ModePermutation(range(300), max_modes=300)
    assert wide.then(wide.inverse()).apply(1 << 299) == (1 << 299, 1)


@pytest.mark.parametrize("destinations", [(0, 0), (1,), (-1, 0), (False, 1), (0, 1.0)])
def test_reject_invalid_permutations(destinations):
    with pytest.raises(ValueError):
        ModePermutation(destinations)


@pytest.mark.parametrize("mask", [-1, 8, 1.0, True])
def test_reject_invalid_basis_masks(mask):
    with pytest.raises(ValueError):
        ModePermutation((0, 1, 2)).apply(mask)


def test_reject_mismatched_registers_and_check_packet_budget_before_expansion():
    with pytest.raises(ValueError, match="equal size"):
        ModePermutation(()).then(ModePermutation((0,)))
    with pytest.raises(TypeError):
        ModePermutation(()).then(())
    with pytest.raises(ValueError, match="width"):
        ModePermutation.from_packets((0, 1), (1,))
    with pytest.raises(ValueError, match="width"):
        ModePermutation.from_packets((0,), repeat(1))
    for width in (0, -1, True, 1.0):
        with pytest.raises(ValueError):
            ModePermutation.from_packets((0,), (width,))
    with pytest.raises(WiringBudgetExceeded, match="max_modes"):
        ModePermutation(repeat(0), max_modes=4)
    with pytest.raises(WiringBudgetExceeded, match="max_modes"):
        relay_mode_permutation((1, 0), modes_per_port=10**100)


@pytest.mark.parametrize("width,limit,consumed", [(2, 11, 3), (2, 8, 3), (2, 3, 1), (1, 7, 4)])
def test_relay_rejects_at_the_first_sender_whose_two_packets_cannot_fit(width, limit, consumed):
    pulled = []

    def senders():
        for i in range(64):
            pulled.append(i)
            yield i

    with pytest.raises(WiringBudgetExceeded, match="max_modes"):
        relay_mode_permutation(senders(), modes_per_port=width, max_modes=limit)
    assert pulled == list(range(consumed))


def test_relay_accepts_exact_boundary_and_preserves_empty_route_and_validation():
    swap = relay_mode_permutation(iter((1, 0)), modes_per_port=2, max_modes=8)
    assert swap.destinations == (4, 5, 2, 3, 0, 1, 6, 7)
    assert relay_mode_permutation(iter(()), modes_per_port=2, max_modes=1).destinations == ()
    with pytest.raises(ValueError, match="form a permutation"):
        relay_mode_permutation(iter((0, 0)), modes_per_port=2, max_modes=8)
