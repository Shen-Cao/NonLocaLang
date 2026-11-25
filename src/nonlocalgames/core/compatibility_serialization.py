"""Versioned exact behavior/resource compatibility task documents."""
import json
from pathlib import Path

from .compatibility_tasks import NetworkCompatibilityTask
from .json_file import read_json
from .resource_serialization import resource_from_dict, resource_to_dict

SCHEMA = "nonlocalgames.network_compatibility_task.v1"


def compatibility_to_dict(task):
    from ..networks.behavior_serialization import behavior_to_dict
    if not isinstance(task, NetworkCompatibilityTask):
        raise TypeError("Expected NetworkCompatibilityTask")
    return {"schema": SCHEMA, "behavior": behavior_to_dict(task.behavior),
            "resource": resource_to_dict(task.resource)}


def compatibility_from_dict(data, *, max_entries=1_000_000, max_scalar_bits=4096, max_scalar_chars=8192):
    from ..networks.behavior_serialization import behavior_from_dict
    if not isinstance(data, dict) or set(data) != {"schema", "behavior", "resource"} or data['schema'] != SCHEMA:
        raise ValueError("Invalid compatibility task fields or schema")
    return NetworkCompatibilityTask(behavior_from_dict(data['behavior'], max_entries=max_entries,
                                    max_scalar_bits=max_scalar_bits, max_scalar_chars=max_scalar_chars),
                                    resource_from_dict(data['resource']))


def dump_compatibility(task, path):
    Path(path).write_text(json.dumps(compatibility_to_dict(task), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_compatibility(path, *, max_file_bytes=64_000_000, max_entries=1_000_000,
                       max_scalar_bits=4096, max_scalar_chars=8192):
    data = read_json(path, max_file_bytes=max_file_bytes, detect_encoding=True)
    return compatibility_from_dict(data, max_entries=max_entries,
                                   max_scalar_bits=max_scalar_bits, max_scalar_chars=max_scalar_chars)
