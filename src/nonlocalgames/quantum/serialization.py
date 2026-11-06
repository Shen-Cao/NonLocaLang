"""Strict numerical serialization of ordinary tensor-product quantum strategies."""
from math import prod
from numbers import Integral

import numpy as np

from .states import validate_dimensions
from .strategy import QuantumStrategy

STRATEGY_SCHEMA = "nonlocalgames.quantum_strategy.v1"


def _matrix_to_dict(matrix):
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def _matrix_from_dict(data, dimension):
    if not isinstance(data, dict) or set(data) != {"real", "imag"}:
        raise ValueError("Numerical matrices require exactly real and imag fields")
    for part in (data["real"], data["imag"]):
        if (not isinstance(part, list) or len(part) != dimension
                or any(not isinstance(row, list) or len(row) != dimension for row in part)):
            raise ValueError("Matrix data does not match the declared square dimension")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for row in part for v in row):
            raise ValueError("Numerical matrix entries must be real JSON numbers")
    try:
        real, imag = np.asarray(data["real"], dtype=float), np.asarray(data["imag"], dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Matrix entries must be representable as float64") from error
    if not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("Numerical matrices must contain only finite numbers")
    return real + 1j * imag


def strategy_to_dict(strategy):
    """Encode a plain strategy; network witnesses require their source-preserving schema."""
    if not isinstance(strategy, QuantumStrategy):
        raise TypeError("Expected a plain QuantumStrategy; preserve network source provenance separately")
    return {"schema": STRATEGY_SCHEMA, "encoding": "complex_float64",
            "dimensions": list(strategy.dimensions), "tolerance": float(strategy.tolerance),
            "state": _matrix_to_dict(strategy.state),
            "measurements": [[[_matrix_to_dict(effect) for effect in povm] for povm in party]
                             for party in strategy.measurements]}


def strategy_from_dict(data, *, max_joint_dimension=1024, max_matrix_entries=1_000_000):
    """Revalidate every decoded matrix, POVM, and tensor dimension numerically.

    Budgets limit dense matrices before conversion. The serialized tolerance
    remains explicit; loading does not turn numerical feasibility into a proof.
    """
    fields = {"schema", "encoding", "dimensions", "tolerance", "state", "measurements"}
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != STRATEGY_SCHEMA:
        raise ValueError("Invalid quantum strategy fields or schema")
    if data["encoding"] != "complex_float64":
        raise ValueError("Unsupported numerical strategy encoding")
    for name, value in (("max_joint_dimension", max_joint_dimension),
                        ("max_matrix_entries", max_matrix_entries)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(data["dimensions"], list):
        raise ValueError("Strategy dimensions must be a JSON list")
    dimensions = validate_dimensions(data["dimensions"])
    size = prod(dimensions)
    if size > max_joint_dimension:
        raise ValueError("Serialized strategy exceeds max_joint_dimension")
    measurements = data["measurements"]
    if not isinstance(measurements, list) or len(measurements) != len(dimensions):
        raise ValueError("Provide one measurement collection per declared party")
    entries = size * size
    for party, dimension in zip(measurements, dimensions):
        if not isinstance(party, list) or not party or any(not isinstance(povm, list) or not povm
                                                        for povm in party):
            raise ValueError("Measurements must be nonempty party/input/outcome lists")
        entries += sum(len(povm) * dimension * dimension for povm in party)
    if entries > max_matrix_entries:
        raise ValueError("Serialized strategy exceeds max_matrix_entries")
    state = _matrix_from_dict(data["state"], size)
    matrices = tuple(tuple(tuple(_matrix_from_dict(effect, dimension) for effect in povm)
                           for povm in party) for party, dimension in zip(measurements, dimensions))
    return QuantumStrategy(state, matrices, data["tolerance"])
