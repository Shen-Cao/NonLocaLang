"""Immutable local checkpoints published after a complete JSON write."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..core.identity import canonical_json
from ..core.json_file import read_json

MANIFEST_BYTES = 128_000
MEMBER_BYTES = 8_000_000
_TEMP_PREFIX = '.nlg-checkpoint-'


def load_checkpoint(path, max_bytes):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular checkpoint file at {path.name}')
    return read_json(path, max_file_bytes=max_bytes)


def publish(path, record, max_bytes):
    """Return True for the winning writer, False when an immutable record already exists."""
    encoded = canonical_json(record, max_bytes=max_bytes-1)
    temporary = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix=_TEMP_PREFIX,
                                suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.write('\n')
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prefix_size(members, steps):
    """Check the immutable prefix up to the highest member observed during this scan."""
    if not members.exists():
        return 0
    if members.is_symlink() or not members.is_dir():
        raise ValueError('Expected a members directory')
    highest = -1
    for path in members.iterdir():
        name = path.name
        if name.startswith(_TEMP_PREFIX) and name.endswith('.tmp'):
            continue
        stem = path.stem
        if (path.suffix != '.json' or not stem.isascii() or not stem.isdecimal()
                or str(int(stem)) != stem or int(stem) > steps or not path.is_file() or path.is_symlink()):
            raise ValueError(f'Unexpected member checkpoint {name!r}')
        highest = max(highest, int(stem))
    # Other writers only extend the prefix. Entries observed out of order
    # during a concurrent directory scan must now have all their predecessors.
    if any(not (members / f'{k}.json').is_file() for k in range(highest + 1)):
        raise ValueError('Missing checkpoint inside weighted CHSH member prefix')
    return highest + 1
