"""Versioned JSON representation; exact rational strings, no executable payloads."""
import json
from pathlib import Path

from .game import Game, Scenario

SCHEMA = "nonlocalgames.finite.v1"


def to_dict(game):
    return {"schema": SCHEMA, "name": game.name,
            "scenario": {"input_sizes": list(game.scenario.input_sizes),
                         "output_sizes": list(game.scenario.output_sizes)},
            "distribution": list(map(str, game.distribution)),
            "payoff": list(map(str, game.payoff))}


def from_dict(data):
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
    return Game(Scenario(**scenario), data["distribution"], data["payoff"], data["name"])


def dump(game, path):
    Path(path).write_text(json.dumps(to_dict(game), indent=2) + "\n", encoding="utf-8")


def load(path):
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
