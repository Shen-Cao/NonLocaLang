"""Independent exact verification of saved network incompatibility events."""
from fractions import Fraction
from math import prod
from numbers import Integral

from ..core.game import rational
from .compatibility import (CompatibilityResult, CompatibilityViolation, _groups,
                            _validate_context, validate_triangle)

VIOLATION_SCHEMA = "nonlocalgames.network_violation.v1"


def violation_to_dict(violation):
    if not isinstance(violation, CompatibilityViolation):
        raise TypeError("Expected a CompatibilityViolation")
    return {"schema": VIOLATION_SCHEMA, "input_profile": list(violation.input_profile),
            "groups": [list(group) for group in violation.groups],
            "group_outputs": [list(values) for values in violation.group_outputs],
            "lhs": str(violation.lhs), "rhs": str(violation.rhs),
            "required_relation": violation.required_relation}


def violation_from_dict(data):
    fields = {"schema", "input_profile", "groups", "group_outputs", "lhs", "rhs", "required_relation"}
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != VIOLATION_SCHEMA:
        raise ValueError("Invalid network violation fields or schema")
    if not isinstance(data["lhs"], str) or not isinstance(data["rhs"], str):
        raise ValueError("Exact certificate values must be rational strings")
    if not isinstance(data["input_profile"], list) or any(not isinstance(data[key], list)
            or any(not isinstance(row, list) for row in data[key]) for key in ("groups", "group_outputs")):
        raise ValueError("Certificate profiles and groups must be lists")
    return CompatibilityViolation(tuple(data["input_profile"]), tuple(map(tuple, data["groups"])),
                                  tuple(map(tuple, data["group_outputs"])), rational(data["lhs"]),
                                  rational(data["rhs"]), data["required_relation"])


def verify_compatibility_violation(network, behavior, method, violation):
    """Recompute one claimed violated condition using exact arithmetic only.

    Accept a CompatibilityViolation or its serialized dictionary. The graph,
    event labels, numerical sides, and inequality direction are all rechecked.
    An unchanged saved value is never accepted without evaluation on behavior.
    """
    _validate_context(network, behavior)
    if isinstance(violation, dict):
        violation = violation_from_dict(violation)
    if not isinstance(violation, CompatibilityViolation):
        raise TypeError("Expected a CompatibilityViolation or serialized certificate")
    scenario = behavior.scenario
    x = tuple(violation.input_profile)
    scenario.input_index(x)
    groups = tuple(tuple(group) for group in violation.groups)
    outcomes = tuple(tuple(a) for a in violation.group_outputs)
    if method == "source_disjoint_independence":
        groups = _groups(network, groups)
        relation = "equal"
    elif method == "triangle_finner":
        validate_triangle(network)
        if groups != ((0,), (1,), (2,)):
            raise ValueError("Triangle event groups must be the three individual parties")
        relation = "less_or_equal"
    else:
        raise ValueError("Unknown compatibility certificate method")
    if len(outcomes) != len(groups) or any(len(a) != len(g) for a, g in zip(outcomes, groups)):
        raise ValueError("Certificate output labels do not match the party groups")
    if any(isinstance(a, bool) or not isinstance(a, Integral) or not 0 <= a < scenario.output_sizes[i]
           for group, output in zip(groups, outcomes) for i, a in zip(group, output)):
        raise ValueError("Certificate output labels are outside their alphabets")
    if violation.required_relation != relation:
        raise ValueError("Certificate comparison direction is invalid for its method")
    marginals = tuple(behavior.marginal(g, x) for g in groups)
    rhs = prod((m[a] for m, a in zip(marginals, outcomes)), start=Fraction(1))
    union = tuple(sorted(i for group in groups for i in group))
    labels = {i: a for group, output in zip(groups, outcomes) for i, a in zip(group, output)}
    joint = behavior.marginal(union, x)[tuple(labels[i] for i in union)]
    lhs = joint if relation == "equal" else joint**2
    if lhs != rational(violation.lhs) or rhs != rational(violation.rhs):
        raise ValueError("Certificate sides differ from the exact behavior evaluation")
    if (relation == "equal" and lhs == rhs) or (relation == "less_or_equal" and lhs <= rhs):
        raise ValueError("The supplied event does not violate the necessary condition")
    verified = CompatibilityViolation(x, groups, outcomes, lhs, rhs, relation)
    assumptions = ("independent sources with the declared incidence",
                   "local operations without communication or extra global randomness",
                   "external inputs independent of sources")
    return CompatibilityResult(method, network, "incompatible", 1, verified, assumptions)
