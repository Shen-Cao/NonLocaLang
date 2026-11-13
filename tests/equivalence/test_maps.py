"""Checked NEW-to-OLD maps retain local dependence under inversion and composition."""
from dataclasses import FrozenInstanceError
from itertools import repeat
import random

import numpy as np
import pytest

from nonlocalgames import Scenario
from nonlocalgames.equivalence.maps import Relabelling, RelabellingBudgetExceeded


def _cycle(max_map_entries=1_000_000):
    return Relabelling(Scenario((2, 3, 1), (3, 2, 4)), (1, 2, 0),
        ((2, 0, 1), (0,), (1, 0)),
        (((1, 0), (0, 1), (1, 0)), ((2, 0, 3, 1),), ((1, 2, 0), (2, 0, 1))),
        max_map_entries=max_map_entries)


def _random_map(source, rng, budget=1_000_000):
    def permutation(size):
        values = list(range(size))
        rng.shuffle(values)
        return tuple(values)
    parties = permutation(source.parties)
    inputs = tuple(permutation(source.input_sizes[i]) for i in parties)
    outputs = tuple(tuple(permutation(source.output_sizes[i]) for _ in range(source.input_sizes[i]))
                    for i in parties)
    return Relabelling(source, parties, inputs, outputs, max_map_entries=budget)


def test_cyclic_party_mapping_and_input_dependent_outputs_have_explicit_direction():
    mapping = _cycle()
    assert mapping.target == Scenario((3, 1, 2), (2, 4, 3))
    assert mapping.source_input((0, 0, 1)) == (0, 2, 0)
    assert mapping.source_output((0, 0, 1), (1, 2, 0)) == (2, 0, 3)
    inverse = mapping.inverse()
    for x in mapping.source.inputs:
        target_x = inverse.source_input(x)
        assert mapping.source_input(target_x) == x
        for a in mapping.source.outputs:
            target_a = inverse.source_output(x, a)
            assert mapping.source_output(target_x, target_a) == a
    assert mapping.then(inverse) == Relabelling(mapping.source)
    assert inverse.then(mapping) == Relabelling(mapping.target)


def test_composition_agrees_with_profile_transport_and_is_associative():
    rng = random.Random(29)
    source = Scenario((2, 3, 2), (3, 2, 2))
    for _ in range(8):
        first = _random_map(source, rng)
        second = _random_map(first.target, rng)
        third = _random_map(second.target, rng)
        composed = first.then(second)
        for x in second.target.inputs:
            middle_x = second.source_input(x)
            assert composed.source_input(x) == first.source_input(middle_x)
            for a in second.target.outputs:
                assert composed.source_output(x, a) == first.source_output(
                    middle_x, second.source_output(x, a))
        assert first.then(second).then(third) == first.then(second.then(third))
        assert composed.inverse() == second.inverse().then(first.inverse())


def test_defaults_are_immutable_identity_maps_and_integrals_are_normalized():
    scenario = Scenario((2, 1), (3, 2))
    identity = Relabelling(scenario)
    for x in scenario.inputs:
        assert identity.source_input(x) == x
        for a in scenario.outputs:
            assert identity.source_output(x, a) == a
    numeric = Relabelling(scenario, (np.int64(0), np.int64(1)))
    assert numeric == identity
    assert all(type(i) is int for i in numeric.party_permutation)
    with pytest.raises(FrozenInstanceError):
        identity.target = scenario
    with pytest.raises(TypeError):
        identity.input_maps[0][0] = 1


def test_map_count_budget_precedes_all_iterable_consumption_and_identity_expansion():
    class Untouched:
        def __iter__(self):
            pytest.fail('budget preflight must not consume map data')
    large = Scenario((10**100,), (2,))
    for options in ({}, {'party_permutation': Untouched()}, {'input_maps': Untouched()},
                    {'output_maps': Untouched()}):
        with pytest.raises(RelabellingBudgetExceeded, match='max_map_entries'):
            Relabelling(large, **options)
    assert Relabelling(Scenario((2,), (3,)), max_map_entries=9).target == Scenario((2,), (3,))
    with pytest.raises(RelabellingBudgetExceeded):
        Relabelling(Scenario((2,), (3,)), max_map_entries=8)


def test_inverse_and_composition_keep_explicit_budget_policies(monkeypatch):
    first = _cycle(max_map_entries=2_000_000)
    assert first == _cycle(max_map_entries=3_000_000)
    assert hash(first) == hash(_cycle(max_map_entries=3_000_000))
    assert first.inverse().max_map_entries == 2_000_000
    second = Relabelling(first.target, max_map_entries=3_000_000)
    assert first.then(second).max_map_entries == 2_000_000
    assert first.then(second, max_map_entries=25).max_map_entries == 25
    import nonlocalgames.equivalence.maps as maps
    monkeypatch.setattr(maps, '_inverse', lambda *a: pytest.fail('budget must fail before inverse expansion'))
    with pytest.raises(RelabellingBudgetExceeded):
        first.inverse(max_map_entries=24)
    with pytest.raises(RelabellingBudgetExceeded):
        first.then(second, max_map_entries=24)


@pytest.mark.parametrize('options', [
    {'party_permutation': (0, 0)}, {'party_permutation': (False, 1)},
    {'party_permutation': (0., 1)}, {'party_permutation': (1,)},
    {'input_maps': ((0, 0), (0, 1))}, {'input_maps': ((0, 1.), (0, 1))},
    {'output_maps': (((0, 1),), ((0, 1), (0, 1)))},
    {'output_maps': (((False, 1), (0, 1)), ((0, 1), (0, 1)))},
    {'max_map_entries': True}, {'max_map_entries': 0},
])
def test_malformed_maps_are_rejected(options):
    with pytest.raises(ValueError):
        Relabelling(Scenario((2, 2), (2, 2)), **options)


def test_unbounded_map_iterables_are_read_only_to_the_shape_boundary():
    with pytest.raises(ValueError, match='number of entries'):
        Relabelling(Scenario((2, 2), (2, 2)), party_permutation=repeat(0))
    with pytest.raises(ValueError, match='number of entries'):
        Relabelling(Scenario((2, 2), (2, 2)), input_maps=(repeat(0), (0, 1)))
    with pytest.raises(ValueError, match='number of entries'):
        Relabelling(Scenario((2,), (2,)), output_maps=((repeat(0), (0, 1)),))
    mapping = _cycle()
    with pytest.raises(ValueError):
        mapping.source_input(repeat(0))
    with pytest.raises(ValueError):
        mapping.source_output((0, 0, 0), repeat(0))


def test_profiles_and_intermediate_interfaces_must_match():
    mapping = _cycle()
    for profile in ((0, 0), (3, 0, 0), (False, 0, 0), (0., 0, 0)):
        with pytest.raises(ValueError):
            mapping.source_input(profile)
    with pytest.raises(ValueError):
        mapping.source_output((0, 0, 0), (0, 0, 3))
    with pytest.raises(ValueError, match='intermediate'):
        mapping.then(Relabelling(mapping.source))
    with pytest.raises(TypeError):
        mapping.then(object())
