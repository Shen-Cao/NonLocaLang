"""Controls for the physical encoding and the hidden relay's information boundary."""
from dataclasses import dataclass
from itertools import product

import pytest

from nonlocalgames.protocols.fermion_reference import (
    ReadoutContext, balanced_strategy_target, reference_target,
)
from nonlocalgames.protocols.graded import ModePermutation
from nonlocalgames.protocols.quadratic_wire import target_id


ABC, ACB, AB, AC, BC = (2, 0, 1), (1, 2, 0), (1, 0, 2), (2, 1, 0), (0, 2, 1)


@pytest.fixture(scope="module")
def targets():
    return {kind: reference_target(statistics=kind)
            for kind in ("fermionic", "ordinary_tensor", "fermionic_even_pair")}


def _branches(target):
    return dict(zip(target.arena.routes, target.behaviors))


def _changed(outs, party):
    return tuple(3-a if i == party and a in (1, 2) else a for i, a in enumerate(outs))


def test_even_packet_control_replays_tensor_and_distinguishes_pair_loops_exactly(targets):
    # These complete canonical targets were exported before the refactor, at 7a146ec.
    prefix = "nonlocalgames.quadratic_wiring_target.v1:"
    assert target_id(targets["fermionic"]) == (
        prefix + "ada7ff10231338c2802693019b58974b1459efb27b53d20fc9af3de331d45a27")
    assert target_id(targets["ordinary_tensor"]) == (
        prefix + "2621922a729552cc8e0c7828a84e2b64c961b8215bfe51efdf0a800d8c77e93b")
    assert targets["fermionic_even_pair"] == targets["ordinary_tensor"]
    single, double = (_branches(targets[k]) for k in ("fermionic", "fermionic_even_pair"))
    for route, count in zip((ABC, ACB, AB, AC, BC), (0, 0, 24, 24, 16)):
        assert sum(p != q for p, q in zip(single[route].probabilities,
                                          double[route].probabilities, strict=True)) == count


@pytest.mark.parametrize("kind,width", [("fermionic", 1), ("fermionic_even_pair", 2)])
def test_each_route_really_moves_the_same_eight_physical_source_terms(monkeypatch, kind, width):
    # A shortcut that just returns the unsigned tensor table would pass an
    # output-only comparison, but would not implement the physical encoding.
    by_map = {}
    original = ModePermutation.apply

    def record(permutation, mask):
        assert len(permutation.destinations) == 6*width
        by_map.setdefault(permutation.destinations, set()).add(mask)
        result = original(permutation, mask)
        if width == 2:
            assert result[1] == 1
        return result

    monkeypatch.setattr(ModePermutation, "apply", record)
    reference_target(statistics=kind)
    # Source i is |10>+|01> or |1100>+|0011>, never a vacuum preparation.
    full = 2**width - 1
    expected = {sum(local << (2*width*i) for i, local in enumerate(states))
                for states in product((full, full << width), repeat=3)}
    assert len(by_map) == 5 and len(expected) == 8
    assert all(actual == expected for actual in by_map.values())


def test_each_local_calibration_is_shared_in_the_right_routes_only(targets):
    # Six directed (receiver, incoming sender) pairs each occur in a triangle
    # and a two-cycle. Each self-return is a separate, coherent local context.
    shared = {(0, 2): (ABC, AC), (1, 0): (ABC, AB), (2, 1): (ABC, BC),
              (0, 1): (ACB, AB), (1, 2): (ACB, BC), (2, 0): (ACB, AC),
              (0, 0): (BC,), (1, 1): (AC,), (2, 2): (AB,)}
    base = _branches(targets["fermionic"])
    checked_views = 0
    for ((party, sender), routes), question in product(shared.items(), range(2)):
        key = ReadoutContext(receiver=party, incoming_sender=sender, question=question)
        changed = _branches(balanced_strategy_target(swap_sign_labels=(key,)))
        for route, branch in changed.items():
            scenario = branch.scenario
            for xi, x in enumerate(scenario.inputs):
                apply_swap = route in routes and x[party] == question
                for ai, a in enumerate(scenario.outputs):
                    origin = scenario.output_index(_changed(a, party)) if apply_swap else ai
                    assert branch.probabilities[64*xi+ai] == base[route].probabilities[64*xi+origin]
        checked_views += 1
    assert checked_views == 18


def test_leaking_the_route_could_spoof_every_reference_cell_but_is_not_a_local_readout(targets):
    # This is a forbidden, route-dependent output relabeling, not a tensor witness.
    illegal = {AB: 1, AC: 2, BC: 1}
    ordinary = _branches(targets["ordinary_tensor"])
    for route, branch in _branches(targets["fermionic"]).items():
        party = illegal.get(route)
        scenario = branch.scenario
        for xi, x in enumerate(scenario.inputs):
            for ai, a in enumerate(scenario.outputs):
                relabel = party is not None and x[party] == 1
                origin = scenario.output_index(_changed(a, party)) if relabel else ai
                assert branch.probabilities[64*xi+ai] == ordinary[route].probabilities[64*xi+origin]
    with pytest.raises(TypeError, match="ReadoutContext"):
        balanced_strategy_target(statistics="ordinary_tensor",
                                 swap_sign_labels=[(route, i, 1) for route, i in illegal.items()])
    with pytest.raises(TypeError):
        ReadoutContext(receiver=1, incoming_sender=0, question=1, outgoing_receiver=0)

    @dataclass(frozen=True, kw_only=True)
    class RouteAware(ReadoutContext):
        outgoing_receiver: int

    with pytest.raises(TypeError, match="ReadoutContext"):
        balanced_strategy_target(swap_sign_labels=[
            RouteAware(receiver=1, incoming_sender=0, question=1, outgoing_receiver=0)])


def test_readout_labels_are_exact_indices_and_cannot_silently_override_each_other():
    for key in (dict(receiver=3, incoming_sender=1, question=0),
                dict(receiver=1, incoming_sender=-1, question=0),
                dict(receiver=1, incoming_sender=1, question=True)):
        with pytest.raises(ValueError):
            ReadoutContext(**key)
    same = ReadoutContext(receiver=0, incoming_sender=2, question=0)
    with pytest.raises(ValueError, match="Duplicate"):
        balanced_strategy_target(swap_sign_labels=[same, same])
