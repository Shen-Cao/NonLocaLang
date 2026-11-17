"""Versioned wire records for a classical wiring witness bound to its arena."""
from hashlib import sha256

from ..core.identity import canonical_json
from ..core.numeric_data import positive_budget
from ..core.scalar_wire import scalar_string
from .classical import ClassicalWiringStrategy
from .serialization import arena_id
from .wiring import WiringArena, WiringBudgetExceeded

_SCHEMA = 'nonlocalgames.classical_wiring_strategy.v1'


def classical_to_dict(strategy):
    if not isinstance(strategy, ClassicalWiringStrategy):
        raise TypeError('Expected a ClassicalWiringStrategy')
    return {
        'schema': _SCHEMA, 'arena_id': arena_id(strategy.arena),
        'keep_sizes': list(strategy.keep_sizes), 'send_sizes': list(strategy.send_sizes),
        'preparations': [list(map(scalar_string, row)) for row in strategy.preparations],
        'responses': [{'party': party, 'sender': sender,
                       'probabilities': list(map(scalar_string, strategy.responses[party, sender]))}
                      for party, sender in sorted(strategy.responses)],
    }


def classical_strategy_id(strategy):
    encoded = canonical_json(classical_to_dict(strategy)).encode('utf-8')
    return _SCHEMA + ':' + sha256(encoded).hexdigest()


def classical_from_dict(arena, data, *, max_entries=1_000_000, max_scalar_bits=4096,
                        max_scalar_chars=8192):
    """Check arena binding, table labels and aggregate sizes before exact decoding."""
    if not isinstance(arena, WiringArena):
        raise TypeError('Expected a WiringArena')
    fields = {'schema', 'arena_id', 'keep_sizes', 'send_sizes', 'preparations', 'responses'}
    if not isinstance(data, dict) or set(data) != fields or data['schema'] != _SCHEMA:
        raise ValueError('Invalid classical wiring strategy fields or schema')
    if data['arena_id'] != arena_id(arena):
        raise ValueError('Classical strategy belongs to a different wiring arena')
    count = arena.scenario.parties
    for key in ('keep_sizes', 'send_sizes'):
        sizes = data[key]
        if (not isinstance(sizes, list) or len(sizes) != count
                or any(type(size) is not int or size < 1 for size in sizes)):
            raise ValueError('Port sizes must be JSON arrays of positive integers, one per party')
    preparations = data['preparations']
    if (not isinstance(preparations, list) or len(preparations) != count
            or any(not isinstance(table, list) for table in preparations)):
        raise ValueError('Provide one JSON preparation array per origin')
    rows = data['responses']
    views = {(party, sender) for route in arena.routes for party, sender in enumerate(route)}
    if not isinstance(rows, list) or len(rows) != len(views):
        raise ValueError('Provide one response record per reachable local view')
    tables = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'party', 'sender', 'probabilities'}:
            raise ValueError('Invalid wiring response record')
        key = row['party'], row['sender']
        if any(type(i) is not int for i in key) or key not in views or key in tables:
            raise ValueError('Invalid or duplicate wiring response view')
        if not isinstance(row['probabilities'], list):
            raise ValueError('Wiring response probabilities must use JSON arrays')
        tables[key] = row['probabilities']
    limit = positive_budget(max_entries, 'max_entries')
    if sum(map(len, preparations)) + sum(map(len, tables.values())) > limit:
        raise WiringBudgetExceeded('Classical wiring strategy exceeds max_entries')
    if any(not isinstance(p, str) for row in (*preparations, *tables.values()) for p in row):
        raise ValueError('Serialized wiring probabilities must be rational strings')
    return ClassicalWiringStrategy(arena, data['keep_sizes'], data['send_sizes'], preparations,
                                   tables, max_entries=limit, max_scalar_bits=max_scalar_bits,
                                   max_scalar_chars=max_scalar_chars)
