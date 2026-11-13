import json

import pytest

from nonlocalgames import Scenario
from nonlocalgames.equivalence.maps import Relabelling, RelabellingBudgetExceeded
from nonlocalgames.equivalence.serialization import (relabelling_from_dict, relabelling_id,
                                                    relabelling_to_dict)


def test_descriptor_roundtrip_binds_exact_maps_but_not_allocation_policy():
    source = Scenario((2, 3), (2, 3))
    mapping = Relabelling(source, (1, 0), ((2, 0, 1), (1, 0)),
                          (((2, 0, 1), (1, 2, 0), (0, 1, 2)), ((1, 0), (0, 1))))
    payload = json.loads(json.dumps(relabelling_to_dict(mapping)))
    restored = relabelling_from_dict(payload, max_map_entries=100)
    assert restored == mapping and restored.target == mapping.target
    assert relabelling_id(restored) == relabelling_id(mapping)
    assert relabelling_id(mapping.inverse()) != relabelling_id(mapping)
    assert 'max_map_entries' not in payload


def test_descriptor_wire_budget_shape_and_integer_types_are_explicit():
    payload = relabelling_to_dict(Relabelling(Scenario((2, 2), (2, 2))))
    with pytest.raises(RelabellingBudgetExceeded):
        relabelling_from_dict(payload, max_map_entries=13)
    with pytest.raises(ValueError, match='max_bytes'):
        relabelling_from_dict(payload, max_bytes=10)
    for bad in (True, 0.0, '0'):
        changed = json.loads(json.dumps(payload))
        changed['party_permutation'][0] = bad
        with pytest.raises((TypeError, ValueError)):
            relabelling_from_dict(changed)
    with pytest.raises(ValueError, match='fields'):
        relabelling_from_dict({**payload, 'guarantee': 'exact'})
    with pytest.raises(ValueError, match='JSON lists'):
        relabelling_from_dict({**payload, 'input_maps': ((0, 1), (0, 1))})
