"""Bounded, atomic storage for the uniform XOR corpus."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..core.numeric_data import positive_budget


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate XOR3 archive key {key!r}")
        result[key] = value
    return result


def _reject_constant(name):
    raise ValueError(f"Invalid JSON constant {name!r}")


def read_archive(path, max_file_bytes):
    limit = positive_budget(max_file_bytes, 'max_file_bytes')
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('XOR3 archive exceeds max_file_bytes')
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except RecursionError as error:
        raise ValueError('XOR3 archive exceeds JSON nesting limit') from error


def write_archive(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
