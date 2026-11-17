"""Bounded, atomic storage for the uniform XOR corpus."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

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
