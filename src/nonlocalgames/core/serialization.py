"""Versioned JSON representation; exact rational strings, no executable payloads."""
import json
from pathlib import Path

from .game import Game, Scenario
from .numeric_data import positive_budget
from .scalar_wire import parse_rational, scalar_string
from .wire_validation import bounded_product

SCHEMA = "nonlocalgames.finite.v1"


def to_dict(game):
    return {"schema": SCHEMA, "name": game.name,
            "scenario": {"input_sizes": list(game.scenario.input_sizes),
                         "output_sizes": list(game.scenario.output_sizes)},
            "distribution": list(map(scalar_string, game.distribution)),
            "payoff": list(map(scalar_string, game.payoff))}


def from_dict(data, *, max_entries=1_000_000, max_scalar_bits=4096, max_scalar_chars=8192):
    if not isinstance(data, dict) or set(data) != {"schema", "name", "scenario",
                                                   "distribution", "payoff"}:
        raise ValueError("Invalid finite game document fields")
    if data["schema"] != SCHEMA:
        raise ValueError("Unsupported game schema")
    scenario = data["scenario"]
    if not isinstance(scenario, dict) or set(scenario) != {"input_sizes", "output_sizes"}:
        raise ValueError("Invalid scenario fields")
    if any(not isinstance(v, list) or any(not isinstance(x, str) for x in v)
           for v in (data["distribution"], data["payoff"])):
        raise ValueError("Serialized rational tables must be lists of strings")
    scenario = Scenario(**scenario)
    inputs = bounded_product(scenario.input_sizes, max_entries, 'max_entries')
    entries = bounded_product(scenario.input_sizes + scenario.output_sizes, max_entries, 'max_entries')
    if len(data['distribution']) != inputs or len(data['payoff']) != entries:
        raise ValueError('Serialized table sizes do not match the scenario')
    limits = dict(max_bits=max_scalar_bits, max_chars=max_scalar_chars)
    distribution = tuple(parse_rational(v, **limits) for v in data['distribution'])
    payoff = tuple(parse_rational(v, **limits) for v in data['payoff'])
    return Game(scenario, distribution, payoff, data['name'])


def dump(game, path):
    Path(path).write_text(json.dumps(to_dict(game), indent=2) + "\n", encoding="utf-8")


def load(path, *, max_file_bytes=64_000_000, **limits):
    positive_budget(max_file_bytes, 'max_file_bytes')
    with Path(path).open('rb') as stream:
        payload = stream.read(max_file_bytes+1)
    if len(payload) > max_file_bytes:
        raise ValueError('Verifier JSON exceeds max_file_bytes')
    return from_dict(json.loads(payload), **limits)
