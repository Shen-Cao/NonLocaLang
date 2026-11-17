"""Versioned descriptions of a hidden wiring arena and its target distributions."""
from hashlib import sha256

from ..core.game import Scenario
from ..core.identity import canonical_json
from ..core.numeric_data import positive_budget
from ..networks.behavior_serialization import behavior_from_dict, behavior_to_dict
from .wiring import (DEFAULT_MAX_PARTIES, DEFAULT_MAX_ROUTES, DEFAULT_MAX_TOTAL_ENTRIES,
                     WiringArena, WiringBudgetExceeded, WiringTarget, _route)

ARENA_SCHEMA = "nonlocalgames.wiring_arena.v1"
TARGET_SCHEMA = "nonlocalgames.wiring_target.v1"


def _semantics():
    return {"preparation": "independent_kept_sent_pairs",
            "relay": "hidden_passive_permutation",
            "route_direction": "receiver_to_sender",
            "question_timing": "after_routing",
            "local_view": "receiver_question_incoming_sender",
            "external_shared_randomness": "forbidden"}


def _scenario_document(scenario):
    return {"input_sizes": list(scenario.input_sizes), "output_sizes": list(scenario.output_sizes)}


def arena_to_dict(arena):
    if not isinstance(arena, WiringArena):
        raise TypeError("Expected WiringArena")
    return {"schema": ARENA_SCHEMA, "semantics": _semantics(),
            "scenario": _scenario_document(arena.scenario), "routes": [list(r) for r in arena.routes]}


def _read_arena(data, max_routes, max_parties, max_total_entries):
    if (not isinstance(data, dict) or set(data) != {"schema", "semantics", "scenario", "routes"}
            or data["schema"] != ARENA_SCHEMA or data["semantics"] != _semantics()):
        raise ValueError("Invalid wiring arena schema, fields, or semantics")
    scenario, routes = data["scenario"], data["routes"]
    if (not isinstance(scenario, dict) or set(scenario) != {"input_sizes", "output_sizes"}
            or any(not isinstance(sizes, list) for sizes in scenario.values())):
        raise ValueError("Wiring scenario must specify input/output alphabet arrays")
    if any(len(sizes) > positive_budget(max_parties, "max_parties") for sizes in scenario.values()):
        raise WiringBudgetExceeded("Wiring scenario exceeds max_parties")
    if not isinstance(routes, list):
        raise ValueError("Wiring routes must be a list")
    if len(routes) > positive_budget(max_routes, "max_routes"):
        raise WiringBudgetExceeded("Wiring arena exceeds max_routes")
    if any(not isinstance(route, list) for route in routes):
        raise ValueError("Each wiring route must be a list of senders")
    return WiringArena(Scenario(**scenario), routes, max_routes=max_routes, max_parties=max_parties,
                       max_total_entries=max_total_entries)


def arena_from_dict(data, *, max_routes=DEFAULT_MAX_ROUTES, max_parties=DEFAULT_MAX_PARTIES,
                    max_total_entries=DEFAULT_MAX_TOTAL_ENTRIES, max_bytes=8_000_000):
    canonical_json(data, max_bytes=max_bytes)
    return _read_arena(data, max_routes, max_parties, max_total_entries)


def target_to_dict(target):
    if not isinstance(target, WiringTarget):
        raise TypeError("Expected WiringTarget")
    branches = [{"route": list(r), "behavior": behavior_to_dict(b)}
                for r, b in zip(target.arena.routes, target.behaviors)]
    return {"schema": TARGET_SCHEMA, "arena": arena_to_dict(target.arena), "branches": branches}


def target_from_dict(data, *, max_routes=DEFAULT_MAX_ROUTES, max_parties=DEFAULT_MAX_PARTIES,
                     max_total_entries=DEFAULT_MAX_TOTAL_ENTRIES, max_scalar_bits=4096,
                     max_scalar_chars=8192, max_bytes=8_000_000):
    canonical_json(data, max_bytes=max_bytes)
    if (not isinstance(data, dict) or set(data) != {"schema", "arena", "branches"}
            or data["schema"] != TARGET_SCHEMA):
        raise ValueError("Invalid wiring target schema or fields")
    arena = _read_arena(data["arena"], max_routes, max_parties, max_total_entries)
    branches = data["branches"]
    if not isinstance(branches, list) or len(branches) != len(arena.routes):
        raise ValueError("Provide one branch object per route")
    scenario = _scenario_document(arena.scenario)
    records, expected = {}, set(arena.routes)
    for branch in branches:
        if (not isinstance(branch, dict) or set(branch) != {"route", "behavior"}
                or not isinstance(branch["route"], list)):
            raise ValueError("Each branch must specify its route and behavior")
        route = _route(branch["route"], arena.scenario.parties)
        if route not in expected or route in records:
            raise ValueError("Target branch contains an unknown or duplicate route")
        behavior = branch["behavior"]
        if not isinstance(behavior, dict) or behavior.get("scenario") != scenario:
            raise ValueError("Branch behavior differs from the arena scenario")
        records[route] = behavior
    by_route = {r: behavior_from_dict(b, max_entries=arena.scenario.entries,
                                      max_scalar_bits=max_scalar_bits, max_scalar_chars=max_scalar_chars)
                for r, b in records.items()}
    return WiringTarget(arena, by_route)


def arena_id(arena):
    """Bind the protocol semantics and the complete set of physical routes."""
    body = canonical_json(arena_to_dict(arena)).encode("utf-8")
    return ARENA_SCHEMA + ":" + sha256(body).hexdigest()


def target_id(target):
    """Bind the exact probabilities at every branch to their respective routes."""
    body = canonical_json(target_to_dict(target)).encode("utf-8")
    return TARGET_SCHEMA + ":" + sha256(body).hexdigest()
