"""Portable rational conditional-behavior documents with exact revalidation."""
from ..core.game import Scenario
from ..core.scalar_wire import parse_rational, scalar_string
from .behavior import RationalBehavior, positive_budget

BEHAVIOR_SCHEMA = "nonlocalgames.rational_behavior.v1"


def behavior_to_dict(behavior):
    if not isinstance(behavior, RationalBehavior):
        raise TypeError("Expected a RationalBehavior")
    return {"schema": BEHAVIOR_SCHEMA, "encoding": "rational",
            "scenario": {"input_sizes": list(behavior.scenario.input_sizes),
                         "output_sizes": list(behavior.scenario.output_sizes)},
            "probabilities": list(map(scalar_string, behavior.probabilities))}


def behavior_from_dict(data, *, max_entries=1_000_000, max_scalar_bits=4096, max_scalar_chars=8192):
    if not isinstance(data, dict) or set(data) != {"schema", "encoding", "scenario", "probabilities"}:
        raise ValueError("Invalid behavior document fields")
    if data["schema"] != BEHAVIOR_SCHEMA or data["encoding"] != "rational":
        raise ValueError("Unsupported behavior schema or encoding")
    scenario = data["scenario"]
    if not isinstance(scenario, dict) or set(scenario) != {"input_sizes", "output_sizes"}:
        raise ValueError("Invalid behavior scenario fields")
    scenario = Scenario(**scenario)
    limit = positive_budget(max_entries, "max_entries")
    if scenario.entries > limit:
        raise ValueError("Behavior table exceeds max_entries")
    values = data["probabilities"]
    if not isinstance(values, list) or len(values) != scenario.entries:
        raise ValueError("Behavior table does not match scenario entries")
    if any(not isinstance(p, str) for p in values):
        raise ValueError("Behavior entries must be rational strings")
    probabilities = tuple(parse_rational(p, max_bits=max_scalar_bits, max_chars=max_scalar_chars)
                          for p in values)
    return RationalBehavior(scenario, probabilities, max_entries=max_entries)


def validate_behavior_scalars(behavior, *, max_entries=1_000_000, max_scalar_bits=4096,
                              max_scalar_chars=8192):
    """Budget semantic probabilities before certificate arithmetic or hashing.

    Direct behavior construction remains unrestricted exact Python arithmetic.
    Certificate admission limits supplied scalar sizes, not all derived products.
    """
    if not isinstance(behavior, RationalBehavior):
        raise TypeError("Expected an exactly validated RationalBehavior")
    limit = positive_budget(max_entries, "max_entries")
    if behavior.scenario.entries > limit:
        raise ValueError("Behavior table exceeds max_entries")
    if len(behavior.probabilities) != behavior.scenario.entries:
        raise ValueError("Behavior table does not match scenario entries")
    for value in behavior.probabilities:
        parse_rational(value, max_bits=max_scalar_bits, max_chars=max_scalar_chars)
