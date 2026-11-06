"""Versioned network structure and witness data; no executable payloads."""
import numpy as np

from ..core.game import Scenario
from .classical import ClassicalNetworkStrategy
from .quantum import QuantumNetworkStrategy, QuantumSource
from .structure import SourceNetwork

NETWORK_SCHEMA = "nonlocalgames.source_network.v1"
CLASSICAL_SCHEMA = "nonlocalgames.classical_network_strategy.v1"
QUANTUM_SCHEMA = "nonlocalgames.quantum_network_strategy.v1"


def _fields(data, fields, schema=None):
    if not isinstance(data, dict) or set(data) != set(fields):
        raise ValueError("Unexpected serialized network fields")
    if schema is not None and data["schema"] != schema:
        raise ValueError("Unsupported network schema")


def network_to_dict(network):
    return {"schema": NETWORK_SCHEMA, "parties": network.parties,
            "sources": [list(source) for source in network.sources]}


def network_from_dict(data):
    """The v1 schema means independent sources and no external global selector."""
    _fields(data, {"schema", "parties", "sources"}, NETWORK_SCHEMA)
    if not isinstance(data["sources"], list) or any(not isinstance(s, list) for s in data["sources"]):
        raise ValueError("Network sources must be lists of endpoint lists")
    return SourceNetwork(data["parties"], data["sources"])


def classical_to_dict(strategy):
    return {"schema": CLASSICAL_SCHEMA, "encoding": "rational",
            "network": network_to_dict(strategy.network),
            "scenario": {"input_sizes": list(strategy.scenario.input_sizes),
                         "output_sizes": list(strategy.scenario.output_sizes)},
            "source_probabilities": [list(map(str, p)) for p in strategy.source_probabilities],
            "responses": [list(map(str, p)) for p in strategy.responses]}


def classical_from_dict(data):
    _fields(data, {"schema", "encoding", "network", "scenario", "source_probabilities", "responses"}, CLASSICAL_SCHEMA)
    if data["encoding"] != "rational":
        raise ValueError("Classical network witnesses require rational encoding")
    _fields(data["scenario"], {"input_sizes", "output_sizes"})
    for key in ("source_probabilities", "responses"):
        if not isinstance(data[key], list) or any(not isinstance(row, list) or
                any(not isinstance(p, str) for p in row) for row in data[key]):
            raise ValueError("Rational tables must be lists of lists of strings")
    return ClassicalNetworkStrategy(network_from_dict(data["network"]), Scenario(**data["scenario"]),
                                    data["source_probabilities"], data["responses"])


def _matrix_to_dict(matrix):
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def _matrix_from_dict(data):
    _fields(data, {"real", "imag"})
    real, imag = np.asarray(data["real"], dtype=float), np.asarray(data["imag"], dtype=float)
    if real.shape != imag.shape or real.ndim != 2:
        raise ValueError("Real and imaginary matrix parts must have identical matrix shapes")
    return real + 1j * imag


def quantum_to_dict(strategy):
    return {"schema": QUANTUM_SCHEMA, "encoding": "complex_float64",
            "network": network_to_dict(strategy.network), "tolerance": strategy.tolerance,
            "sources": [{"dimensions": list(s.dimensions), "tolerance": s.tolerance,
                         "state": _matrix_to_dict(s.state)} for s in strategy.sources],
            "measurements": [[[_matrix_to_dict(effect) for effect in povm] for povm in party]
                             for party in strategy.measurements]}


def quantum_from_dict(data, *, max_total_dimension=1024):
    """Reconstruct and numerically revalidate every source and local POVM."""
    _fields(data, {"schema", "encoding", "network", "tolerance", "sources", "measurements"}, QUANTUM_SCHEMA)
    if data["encoding"] != "complex_float64":
        raise ValueError("Unsupported quantum network numerical encoding")
    if not isinstance(data["sources"], list) or not isinstance(data["measurements"], list):
        raise ValueError("Sources and measurements must be lists")
    sources = []
    for source in data["sources"]:
        _fields(source, {"dimensions", "tolerance", "state"})
        sources.append(QuantumSource(_matrix_from_dict(source["state"]), source["dimensions"], source["tolerance"]))
    measurements = []
    for party in data["measurements"]:
        if not isinstance(party, list) or any(not isinstance(povm, list) for povm in party):
            raise ValueError("Measurements must be nested party/input/outcome lists")
        measurements.append(tuple(tuple(_matrix_from_dict(effect) for effect in povm) for povm in party))
    return QuantumNetworkStrategy(network_from_dict(data["network"]), sources, measurements,
                                  data["tolerance"], max_total_dimension)
