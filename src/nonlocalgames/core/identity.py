"""Exact content identities, distinct from score or relabelling equivalence."""
import hashlib
import json
from numbers import Integral

from .scalar_wire import scalar_string

VERIFIER_ID_PREFIX = "nonlocalgames.verifier.v1:"
RESOURCE_ID_PREFIX = "nonlocalgames.resource.v2:"


def _json_values(data, active):
    if data is None or isinstance(data, (str, bool, int, float)):
        return
    if not isinstance(data, (dict, list, tuple)):
        raise TypeError("Canonical JSON accepts only JSON scalars, string-keyed objects, and arrays")
    identifier = id(data)
    if identifier in active:
        raise ValueError("Canonical JSON does not accept cyclic data")
    active.add(identifier)
    try:
        if isinstance(data, dict):
            if any(not isinstance(key, str) for key in data):
                raise TypeError("Canonical JSON object keys must be strings")
            values = data.values()
        else:
            values = data
        for value in values:
            _json_values(value, active)
    finally:
        active.remove(identifier)


def canonical_json(data, *, max_bytes=None):
    """Encode deterministic JSON without custom conversions or nonfinite numbers.

    Tuples encode as JSON arrays. No rational-to-string conversion is implicit:
    exact coefficient payloads must already contain their intended wire strings.
    An optional byte budget stops accumulation before retaining an oversized
    encoded document. It does not promise a wall-clock or Python-object budget.
    """
    if max_bytes is not None and (isinstance(max_bytes, bool)
            or not isinstance(max_bytes, Integral) or max_bytes < 1):
        raise ValueError("max_bytes must be a positive integer")
    try:
        _json_values(data, set())
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
        chunks, size = [], 0
        for chunk in encoder.iterencode(data):
            size += len(chunk.encode("utf-8"))
            if max_bytes is not None and size > max_bytes:
                raise ValueError("Canonical JSON exceeds max_bytes")
            chunks.append(chunk)
        return "".join(chunks)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds serialization limits") from error


def verifier_digest(game):
    """Preserve the original XOR certificate digest byte for byte; names are excluded."""
    from .game import Game
    if not isinstance(game, Game):
        raise TypeError("Expected a finite Game")
    payload = {"input_sizes": game.scenario.input_sizes,
               "output_sizes": game.scenario.output_sizes,
               "distribution": tuple(map(scalar_string, game.distribution)),
               "payoff": tuple(map(scalar_string, game.payoff))}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def verifier_id(game):
    return VERIFIER_ID_PREFIX + verifier_digest(game)


def resource_id(spec):
    """Hash the exact current resource-codec representation, with no approximate quotient."""
    from .resources import ResourceSpec
    from .resource_serialization import resource_to_dict
    if not isinstance(spec, ResourceSpec):
        raise TypeError("Expected a ResourceSpec")
    return RESOURCE_ID_PREFIX + hashlib.sha256(
        canonical_json(resource_to_dict(spec)).encode("utf-8")).hexdigest()
