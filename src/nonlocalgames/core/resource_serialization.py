"""Versioned resource/task data with explicit numerical fixed-state payloads."""
import json
from pathlib import Path

from .resources import AnalysisTask, ResourceSpec
from .serialization import from_dict as game_from_dict, to_dict as game_to_dict

RESOURCE_SCHEMA = "nonlocalgames.resource.v2"
TASK_SCHEMA = "nonlocalgames.task.v1"


def resource_to_dict(resource):
    fixed = resource.fixed_state
    state = None if fixed is None else {"encoding": "complex_float64",
            "dimensions": list(fixed.dimensions), "tolerance": fixed.tolerance,
            "real": fixed.state.real.tolist(), "imag": fixed.state.imag.tolist()}
    from ..networks.serialization import network_to_dict
    return {"schema": RESOURCE_SCHEMA, "kind": resource.kind.value,
            "network": None if resource.network is None else network_to_dict(resource.network),
            "local_dimensions": None if resource.local_dimensions is None else list(resource.local_dimensions),
            "shared_randomness": resource.shared_randomness.value,
            "state_constraint": resource.state_constraint.value,
            "measurement_class": resource.measurement_class.value,
            "ppt_cuts": [list(cut) for cut in resource.ppt_cuts], "fixed_state": state}


def resource_from_dict(data):
    fields = {"schema", "kind", "local_dimensions", "shared_randomness", "state_constraint",
              "measurement_class", "ppt_cuts", "fixed_state"}
    if isinstance(data, dict) and data.get("schema") == "nonlocalgames.resource.v1" and set(data) == fields:
        data = {**data, "schema": RESOURCE_SCHEMA, "network": None}
    fields.add("network")
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != RESOURCE_SCHEMA:
        raise ValueError("Invalid resource document fields or schema")
    fixed = data["fixed_state"]
    if fixed is not None:
        import numpy as np
        from ..quantum.states import FixedState
        if not isinstance(fixed, dict) or set(fixed) != {"encoding", "dimensions", "tolerance", "real", "imag"}:
            raise ValueError("Invalid numerical fixed-state fields")
        if fixed["encoding"] != "complex_float64":
            raise ValueError("Unsupported fixed-state numerical encoding")
        real, imag = np.array(fixed["real"], dtype=float), np.array(fixed["imag"], dtype=float)
        if real.shape != imag.shape:
            raise ValueError("Fixed-state real and imaginary parts must have identical shapes")
        fixed = FixedState(real + 1j * imag, fixed["dimensions"], fixed["tolerance"])
    from ..networks.serialization import network_from_dict
    network = None if data["network"] is None else network_from_dict(data["network"])
    return ResourceSpec(data["kind"], data["local_dimensions"], network=network,
                        shared_randomness=data["shared_randomness"], state_constraint=data["state_constraint"],
                        measurement_class=data["measurement_class"], ppt_cuts=data["ppt_cuts"], fixed_state=fixed)


def task_to_dict(task):
    return {"schema": TASK_SCHEMA, "game": game_to_dict(task.game),
            "resource": resource_to_dict(task.resource)}


def task_from_dict(data):
    if not isinstance(data, dict) or set(data) != {"schema", "game", "resource"} or data["schema"] != TASK_SCHEMA:
        raise ValueError("Invalid task document fields or schema")
    return AnalysisTask(game_from_dict(data["game"]), resource_from_dict(data["resource"]))


def dump_task(task, path):
    Path(path).write_text(json.dumps(task_to_dict(task), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_task(path):
    return task_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
