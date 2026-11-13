"""Portable local-bijection descriptors; a descriptor is not an evidence artifact."""
from hashlib import sha256

from ..core.game import Scenario
from ..core.identity import canonical_json
from .maps import Relabelling

RELABEL_SCHEMA = 'nonlocalgames.relabelling.v1'


def relabelling_to_dict(relabelling):
    if not isinstance(relabelling, Relabelling):
        raise TypeError('Expected Relabelling')
    return {'schema': RELABEL_SCHEMA,
            'source': {'input_sizes': list(relabelling.source.input_sizes),
                       'output_sizes': list(relabelling.source.output_sizes)},
            'party_permutation': list(relabelling.party_permutation),
            'input_maps': [list(row) for row in relabelling.input_maps],
            'output_maps': [[list(row) for row in party] for party in relabelling.output_maps]}


def relabelling_from_dict(data, *, max_map_entries=1_000_000, max_bytes=8_000_000):
    canonical_json(data, max_bytes=max_bytes)
    fields = {'schema', 'source', 'party_permutation', 'input_maps', 'output_maps'}
    if not isinstance(data, dict) or set(data) != fields or data['schema'] != RELABEL_SCHEMA:
        raise ValueError('Invalid relabelling fields or schema')
    source = data['source']
    if (not isinstance(source, dict) or set(source) != {'input_sizes', 'output_sizes'}
            or any(not isinstance(source[key], list) for key in source)):
        raise ValueError('Relabelling source must be a scenario with JSON-list alphabets')
    if (not isinstance(data['party_permutation'], list)
            or not isinstance(data['input_maps'], list)
            or any(not isinstance(row, list) for row in data['input_maps'])
            or not isinstance(data['output_maps'], list)
            or any(not isinstance(party, list) or any(not isinstance(row, list) for row in party)
                   for party in data['output_maps'])):
        raise ValueError('Relabelling maps must be JSON lists')
    return Relabelling(Scenario(**source), data['party_permutation'], data['input_maps'],
                       data['output_maps'], max_map_entries=max_map_entries)


def relabelling_id(relabelling):
    payload = canonical_json(relabelling_to_dict(relabelling))
    return RELABEL_SCHEMA + ':' + sha256(payload.encode('utf-8')).hexdigest()
