"""Decode JSON without duplicate fields; bound file bytes before parsing."""
import json
from pathlib import Path

from .numeric_data import positive_budget


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(name):
    raise ValueError(f"Invalid JSON constant {name!r}")


def decode_json(source):
    """Decode an already size-checked source without discarding repeated keys."""
    try:
        return json.loads(source, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except RecursionError as error:
        raise ValueError('JSON document exceeds JSON nesting limit') from error


def read_json(path, *, max_file_bytes=8_000_000, detect_encoding=False):
    """Read UTF-8 JSON, or preserve stdlib BOM/UTF-16/32 detection for older codecs."""
    limit = positive_budget(max_file_bytes, 'max_file_bytes')
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('JSON document exceeds max_file_bytes')
    return decode_json(data if detect_encoding else data.decode('utf-8'))
