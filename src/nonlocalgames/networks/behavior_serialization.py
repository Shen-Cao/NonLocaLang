"""Portable rational conditional-behavior documents with exact revalidation."""
from ..core.game import Scenario
from .behavior import RationalBehavior

BEHAVIOR_SCHEMA = "nonlocalgames.rational_behavior.v1"


def behavior_to_dict(behavior):
    if not isinstance(behavior, RationalBehavior):
        raise TypeError("Expected a RationalBehavior")
    return {"schema": BEHAVIOR_SCHEMA, "encoding": "rational",
            "scenario": {"input_sizes": list(behavior.scenario.input_sizes),
                         "output_sizes": list(behavior.scenario.output_sizes)},
            "probabilities": list(map(str, behavior.probabilities))}


def behavior_from_dict(data, *, max_entries=1_000_000):
    if not isinstance(data, dict) or set(data) != {"schema", "encoding", "scenario", "probabilities"}:
        raise ValueError("Invalid behavior document fields")
    if data["schema"] != BEHAVIOR_SCHEMA or data["encoding"] != "rational":
        raise ValueError("Unsupported behavior schema or encoding")
    scenario = data["scenario"]
    if not isinstance(scenario, dict) or set(scenario) != {"input_sizes", "output_sizes"}:
        raise ValueError("Invalid behavior scenario fields")
    if not isinstance(data["probabilities"], list) or any(not isinstance(p, str) for p in data["probabilities"]):
        raise ValueError("Behavior entries must be rational strings")
    return RationalBehavior(Scenario(**scenario), data["probabilities"], max_entries=max_entries)
