"""Portable wiring targets with exact probabilities in the positive real embedding of Q(√2)."""
from hashlib import sha256

from ..core.identity import canonical_json
from .quadratic import scalar_from_pair, scalar_to_pair
from .quadratic_behavior import QuadraticBehavior, QuadraticWiringTarget
from .serialization import arena_from_dict, arena_to_dict
from .wiring import DEFAULT_MAX_PARTIES, DEFAULT_MAX_ROUTES, DEFAULT_MAX_TOTAL_ENTRIES, _route

TARGET_SCHEMA = "nonlocalgames.quadratic_wiring_target.v1"
_ENCODING = "a_plus_b_sqrt2_positive"


def target_to_dict(target):
    if not isinstance(target, QuadraticWiringTarget):
        raise TypeError("Expected QuadraticWiringTarget")
    branches = [{"route": list(route), "probabilities": [scalar_to_pair(p) for p in branch.probabilities]}
                for route, branch in zip(target.arena.routes, target.behaviors)]
    return {"schema": TARGET_SCHEMA, "encoding": _ENCODING,
            "arena": arena_to_dict(target.arena), "branches": branches}


def target_from_dict(data, *, max_routes=DEFAULT_MAX_ROUTES, max_parties=DEFAULT_MAX_PARTIES,
                     max_total_entries=DEFAULT_MAX_TOTAL_ENTRIES, max_scalar_bits=4096,
                     max_scalar_chars=8192, max_bytes=8_000_000):
    canonical_json(data, max_bytes=max_bytes)
    if (not isinstance(data, dict) or set(data) != {"schema", "encoding", "arena", "branches"}
            or data["schema"] != TARGET_SCHEMA or data["encoding"] != _ENCODING):
        raise ValueError("Invalid quadratic wiring target schema, fields, or encoding")
    arena = arena_from_dict(data["arena"], max_routes=max_routes, max_parties=max_parties,
                            max_total_entries=max_total_entries, max_bytes=max_bytes)
    rows = data["branches"]
    if not isinstance(rows, list) or len(rows) != len(arena.routes):
        raise ValueError("Provide one quadratic branch per route")
    expected, tables = set(arena.routes), {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"route", "probabilities"} or not isinstance(row["route"], list):
            raise ValueError("Quadratic branches require a route and probabilities")
        route = _route(row["route"], arena.scenario.parties)
        if route not in expected or route in tables:
            raise ValueError("Unknown or duplicate quadratic branch route")
        values = row["probabilities"]
        if not isinstance(values, list) or len(values) != arena.scenario.entries:
            raise ValueError("Quadratic branch probability table differs from the scenario")
        tables[route] = values
    by_route = {}
    for route, values in tables.items():
        scalars = (scalar_from_pair(p, max_scalar_bits=max_scalar_bits, max_scalar_chars=max_scalar_chars)
                   for p in values)
        by_route[route] = QuadraticBehavior(arena.scenario, scalars, max_entries=arena.scenario.entries,
                                             max_scalar_bits=max_scalar_bits, max_parties=max_parties)
    return QuadraticWiringTarget(arena, by_route)


def target_id(target):
    body = canonical_json(target_to_dict(target)).encode("utf-8")
    return TARGET_SCHEMA + ":" + sha256(body).hexdigest()
