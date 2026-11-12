"""Versioned resource/task data with explicit numerical fixed-state payloads."""
import json
from math import prod
from pathlib import Path

from .resources import AnalysisTask, ResourceSpec
from .numeric_data import matrix_from_dict, positive_budget
from .serialization import from_dict as game_from_dict, to_dict as game_to_dict

RESOURCE_SCHEMA = "nonlocalgames.resource.v2"
TASK_SCHEMA = "nonlocalgames.task.v1"


def resource_to_dict(resource):
    fixed = resource.fixed_state
    state = None if fixed is None else {"encoding": "complex_float64",
            "dimensions": list(fixed.dimensions), "tolerance": float(fixed.tolerance),
            "real": fixed.state.real.tolist(), "imag": fixed.state.imag.tolist()}
    network = None
    if resource.network is not None:
        from ..networks.serialization import network_to_dict
        network = network_to_dict(resource.network)
    return {"schema": RESOURCE_SCHEMA, "kind": resource.kind.value,
            "network": network,
            "local_dimensions": None if resource.local_dimensions is None else list(resource.local_dimensions),
            "shared_randomness": resource.shared_randomness.value,
            "state_constraint": resource.state_constraint.value,
            "measurement_class": resource.measurement_class.value,
            "ppt_cuts": [list(cut) for cut in resource.ppt_cuts], "fixed_state": state}


def resource_from_dict(data, *, max_joint_dimension=1024, max_matrix_entries=1_000_000):
    positive_budget(max_joint_dimension, "max_joint_dimension")
    positive_budget(max_matrix_entries, "max_matrix_entries")
    fields = {"schema", "kind", "local_dimensions", "shared_randomness", "state_constraint",
              "measurement_class", "ppt_cuts", "fixed_state"}
    if isinstance(data, dict) and data.get("schema") == "nonlocalgames.resource.v1" and set(data) == fields:
        data = {**data, "schema": RESOURCE_SCHEMA, "network": None}
    fields.add("network")
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != RESOURCE_SCHEMA:
        raise ValueError("Invalid resource document fields or schema")
    fixed = data["fixed_state"]
    if fixed is not None:
        from ..quantum.states import FixedState, validate_dimensions
        if not isinstance(fixed, dict) or set(fixed) != {"encoding", "dimensions", "tolerance", "real", "imag"}:
            raise ValueError("Invalid numerical fixed-state fields")
        if fixed["encoding"] != "complex_float64":
            raise ValueError("Unsupported fixed-state numerical encoding")
        dimensions = validate_dimensions(fixed["dimensions"])
        size = prod(dimensions)
        if size > max_joint_dimension:
            raise ValueError("Fixed state exceeds max_joint_dimension")
        if size * size > max_matrix_entries:
            raise ValueError("Fixed state exceeds max_matrix_entries")
        matrix = matrix_from_dict({"real": fixed["real"], "imag": fixed["imag"]}, size)
        fixed = FixedState(matrix, dimensions, fixed["tolerance"])
    network = None
    if data["network"] is not None:
        from ..networks.serialization import network_from_dict
        network = network_from_dict(data["network"])
    return ResourceSpec(data["kind"], data["local_dimensions"], network=network,
                        shared_randomness=data["shared_randomness"], state_constraint=data["state_constraint"],
                        measurement_class=data["measurement_class"], ppt_cuts=data["ppt_cuts"], fixed_state=fixed)


def task_to_dict(task):
    return {"schema": TASK_SCHEMA, "game": game_to_dict(task.game),
            "resource": resource_to_dict(task.resource)}


def task_from_dict(data, **resource_limits):
    if not isinstance(data, dict) or set(data) != {"schema", "game", "resource"} or data["schema"] != TASK_SCHEMA:
        raise ValueError("Invalid task document fields or schema")
    return AnalysisTask(game_from_dict(data["game"]), resource_from_dict(data["resource"], **resource_limits))


def dump_task(task, path):
    Path(path).write_text(json.dumps(task_to_dict(task), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_task(path, *, max_file_bytes=64_000_000, **resource_limits):
    """Load bounded JSON and decode fixed-state matrices under explicit budgets."""
    positive_budget(max_file_bytes, "max_file_bytes")
    with Path(path).open("rb") as stream:
        payload = stream.read(max_file_bytes + 1)
    if len(payload) > max_file_bytes:
        raise ValueError("Task JSON exceeds max_file_bytes")
    return task_from_dict(json.loads(payload), **resource_limits)
