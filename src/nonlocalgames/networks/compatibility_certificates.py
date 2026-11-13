"""Independent exact verification of saved network incompatibility events."""
from fractions import Fraction
from itertools import islice
from math import prod
from numbers import Integral

from ..core.scalar_wire import parse_rational, scalar_string
from .behavior_serialization import validate_behavior_scalars
from .compatibility import (CompatibilityResult, CompatibilityViolation, _groups,
                            _validate_context, validate_triangle)

VIOLATION_SCHEMA = "nonlocalgames.network_violation.v1"


def violation_to_dict(violation):
    if not isinstance(violation, CompatibilityViolation):
        raise TypeError("Expected a CompatibilityViolation")
    return {"schema": VIOLATION_SCHEMA, "input_profile": list(violation.input_profile),
            "groups": [list(group) for group in violation.groups],
            "group_outputs": [list(values) for values in violation.group_outputs],
            "lhs": scalar_string(violation.lhs), "rhs": scalar_string(violation.rhs),
            "required_relation": violation.required_relation}


def _profile_shape(data, party_count=None):
    profile, groups, outputs = data['input_profile'], data['groups'], data['group_outputs']
    if not isinstance(profile, list) or not profile:
        raise ValueError("Certificate input profile must be a nonempty list")
    count = len(profile) if party_count is None else party_count
    if len(profile) != count:
        raise ValueError("Certificate input profile has the wrong number of parties")
    if (not isinstance(groups, list) or not 2 <= len(groups) <= count
            or any(not isinstance(group, list) or not group for group in groups)
            or sum(map(len, groups)) > count):
        raise ValueError("Certificate group counts do not fit the party profile")
    if (not isinstance(outputs, list) or len(outputs) != len(groups)
            or any(not isinstance(values, list) or len(values) != len(group)
                   for group, values in zip(groups, outputs))):
        raise ValueError("Certificate output profiles do not match its groups")


def violation_from_dict(data, *, max_scalar_bits=4096, max_scalar_chars=8192):
    fields = {"schema", "input_profile", "groups", "group_outputs", "lhs", "rhs", "required_relation"}
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != VIOLATION_SCHEMA:
        raise ValueError("Invalid network violation fields or schema")
    if not isinstance(data["lhs"], str) or not isinstance(data["rhs"], str):
        raise ValueError("Exact certificate values must be rational strings")
    _profile_shape(data)
    return CompatibilityViolation(tuple(data["input_profile"]), tuple(map(tuple, data["groups"])),
                                  tuple(map(tuple, data["group_outputs"])), parse_rational(data["lhs"], max_bits=max_scalar_bits, max_chars=max_scalar_chars),
                                  parse_rational(data["rhs"], max_bits=max_scalar_bits, max_chars=max_scalar_chars), data["required_relation"])


def verify_compatibility_violation(network, behavior, method, violation, *, max_entries=1_000_000,
                                   max_scalar_bits=4096, max_scalar_chars=8192):
    """Recompute one claimed violated condition using exact arithmetic only.

    Accept a CompatibilityViolation or its serialized dictionary. The graph,
    event labels, numerical sides, and inequality direction are all rechecked.
    An unchanged saved value is never accepted without evaluation on behavior.
    """
    _validate_context(network, behavior)
    validate_behavior_scalars(behavior, max_entries=max_entries, max_scalar_bits=max_scalar_bits,
                              max_scalar_chars=max_scalar_chars)
    if isinstance(violation, dict):
        if all(field in violation for field in ('input_profile', 'groups', 'group_outputs')):
            _profile_shape(violation, behavior.scenario.parties)
        violation = violation_from_dict(violation, max_scalar_bits=max_scalar_bits,
                                        max_scalar_chars=max_scalar_chars)
    if not isinstance(violation, CompatibilityViolation):
        raise TypeError("Expected a CompatibilityViolation or serialized certificate")
    scenario = behavior.scenario
    x = tuple(islice(iter(violation.input_profile), scenario.parties+1))
    scenario.input_index(x)
    groups = tuple(tuple(islice(iter(group), scenario.parties+1))
                   for group in islice(iter(violation.groups), scenario.parties+1))
    outcomes = tuple(tuple(islice(iter(a), scenario.parties+1))
                     for a in islice(iter(violation.group_outputs), scenario.parties+1))
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
    claimed_lhs = parse_rational(violation.lhs, max_bits=max_scalar_bits, max_chars=max_scalar_chars)
    claimed_rhs = parse_rational(violation.rhs, max_bits=max_scalar_bits, max_chars=max_scalar_chars)
    marginals = tuple(behavior.marginal(g, x) for g in groups)
    rhs = prod((m[a] for m, a in zip(marginals, outcomes)), start=Fraction(1))
    union = tuple(sorted(i for group in groups for i in group))
    labels = {i: a for group, output in zip(groups, outcomes) for i, a in zip(group, output)}
    joint = behavior.marginal(union, x)[tuple(labels[i] for i in union)]
    lhs = joint if relation == "equal" else joint**2
    if lhs != claimed_lhs or rhs != claimed_rhs:
        raise ValueError("Certificate sides differ from the exact behavior evaluation")
    if (relation == "equal" and lhs == rhs) or (relation == "less_or_equal" and lhs <= rhs):
        raise ValueError("The supplied event does not violate the necessary condition")
    verified = CompatibilityViolation(x, groups, outcomes, lhs, rhs, relation)
    assumptions = ("independent sources with the declared incidence",
                   "local operations without communication or extra global randomness",
                   "external inputs independent of sources")
    return CompatibilityResult(method, network, "incompatible", 1, verified, assumptions)
