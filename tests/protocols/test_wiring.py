from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
from itertools import repeat

import numpy as np
import pytest

from nonlocalgames import Scenario
from nonlocalgames.core.scalar_wire import ScalarBudgetExceeded
from nonlocalgames.networks.behavior import RationalBehavior
from nonlocalgames.protocols.serialization import (
    arena_from_dict, arena_id, arena_to_dict, target_from_dict, target_id, target_to_dict)
from nonlocalgames.protocols.wiring import WiringArena, WiringBudgetExceeded, WiringTarget


ABC = (2, 0, 1)
AC = (2, 1, 0)
SCENARIO = Scenario((2, 1, 1), (2, 1, 1))


def _target():
    arena = WiringArena(SCENARIO, [AC, ABC])
    always_zero = RationalBehavior(SCENARIO, (1, 0, 1, 0))
    copy_input = RationalBehavior(SCENARIO, (1, 0, 0, 1))
    return WiringTarget(arena, {AC: copy_input, ABC: always_zero})


def test_ports_persist_across_contexts_with_distinct_roles_at_self_return():
    arena = _target().arena
    assert arena.routes == (ABC, AC)
    assert arena.held_ports(AC, 1) == ((1, "kept"), (1, "sent"))
    assert arena.held_ports(AC, 0) == arena.held_ports(ABC, 0) == ((0, "kept"), (2, "sent"))
    assert arena.local_view(AC, 0, 1) == arena.local_view(ABC, 0, 1) == (0, 1, 2)
    assert arena.local_view(AC, 1, 0) != arena.local_view(ABC, 1, 0)
    with pytest.raises(ValueError, match="Question"):
        arena.local_view(AC, 0, 2)
    with pytest.raises(ValueError, match="Receiver"):
        arena.held_ports(AC, True)
    with pytest.raises(ValueError, match="belong"):
        arena.local_view((0, 1, 2), 0, 0)


def test_route_order_input_types_and_wire_branch_order_do_not_change_identity():
    target = _target()
    arena = WiringArena(SCENARIO, [tuple(np.int64(x) for x in r) for r in (ABC, AC)])
    assert arena_id(arena) == arena_id(target.arena)
    assert arena_from_dict(json.loads(json.dumps(arena_to_dict(arena)))) == arena
    payload = json.loads(json.dumps(target_to_dict(target)))
    payload["arena"]["routes"].reverse()
    payload["branches"].reverse()
    restored = target_from_dict(payload)
    assert restored == target and target_id(restored) == target_id(target)
    swap = WiringTarget(arena, dict(zip(arena.routes, reversed(target.behaviors))))
    assert target_id(swap) != target_id(target)
    reduced = WiringArena(SCENARIO, (ABC,))
    assert arena_id(reduced) != arena_id(arena)


def test_target_snapshot_and_configuration_are_immutable():
    target = _target()
    branches = dict(zip(target.arena.routes, target.behaviors))
    snapshot = WiringTarget(target.arena, branches)
    branches[ABC] = target.behaviors[1]
    assert snapshot == target
    with pytest.raises(FrozenInstanceError):
        target.arena.routes = ()
    with pytest.raises(FrozenInstanceError):
        target.behaviors = ()


@pytest.mark.parametrize("routes", [[], [ABC, ABC], [(0, 0, 2)], [(0, 1, True)],
                                    [(0, 1, 0.)], [(0, -1, 1)], [(0, 1)], [(0, 1, 3)]])
def test_arena_rejects_invalid_or_ambiguous_wirings(routes):
    with pytest.raises(ValueError):
        WiringArena(SCENARIO, routes)


def test_target_validates_route_keys_and_exact_interfaces():
    target = _target()
    for bindings in ({ABC: target.behaviors[0]}, {(0, 1, 2): target.behaviors[0], AC: target.behaviors[1]},
                     {(2, False, 1): target.behaviors[0], AC: target.behaviors[1]}):
        with pytest.raises(ValueError):
            WiringTarget(target.arena, bindings)
    other = RationalBehavior(Scenario((1, 1, 1), (2, 1, 1)), (1, 0))
    with pytest.raises(ValueError, match="scenario"):
        WiringTarget(target.arena, {ABC: other, AC: target.behaviors[1]})
    with pytest.raises(TypeError, match="keyed"):
        WiringTarget(target.arena, target.behaviors)


def test_dimension_and_entry_guards_precede_route_and_behavior_work(monkeypatch):
    def untouched():
        pytest.fail("must reject before consuming routes")
        yield
    with pytest.raises(WiringBudgetExceeded, match="max_parties"):
        WiringArena(SCENARIO, untouched(), max_parties=2)
    with pytest.raises(WiringBudgetExceeded, match="max_total_entries"):
        WiringArena(SCENARIO, untouched(), max_total_entries=3)
    with pytest.raises(WiringBudgetExceeded, match="max_routes"):
        WiringArena(SCENARIO, repeat(ABC), max_routes=1)
    with pytest.raises(ValueError, match="sender"):
        WiringArena(SCENARIO, [repeat(0)])
    raw = target_to_dict(_target())
    import nonlocalgames.protocols.serialization as codec
    monkeypatch.setattr(codec, "behavior_from_dict", lambda *a, **k: pytest.fail("check arena first"))
    for kwargs in ({"max_total_entries": 7}, {"max_parties": 2}, {"max_routes": 1}):
        with pytest.raises(WiringBudgetExceeded):
            target_from_dict(raw, **kwargs)
    raw["branches"][-1]["route"] = list(ABC)
    with pytest.raises(ValueError, match="duplicate"):
        target_from_dict(raw)


def test_wire_semantics_are_fixed_and_semantic_scalars_are_bounded():
    data = target_to_dict(_target())
    for field, value in (("relay", "public_permutation"), ("question_timing", "before_routing"),
                         ("local_view", "global_route"), ("external_shared_randomness", "allowed")):
        broken = deepcopy(data)
        broken["arena"]["semantics"][field] = value
        with pytest.raises(ValueError, match="semantics"):
            target_from_dict(broken)
    with pytest.raises(ValueError, match="max_bytes"):
        target_from_dict(data, max_bytes=100)
    data["branches"][0]["behavior"]["probabilities"] = ["1e1000000000", "0", "1", "0"]
    with pytest.raises(ScalarBudgetExceeded):
        target_from_dict(data)
    data["branches"][0]["behavior"]["probabilities"] = ["1/4", "3/4", "1", "0"]
    with pytest.raises(ScalarBudgetExceeded):
        target_from_dict(data, max_scalar_bits=2)
    assert target_from_dict(data).behaviors[0].probabilities[0] == Fraction(1, 4)
