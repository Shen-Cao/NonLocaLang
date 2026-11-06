"""Versioned network structure and witness data; no executable payloads."""
from math import prod

from ..core.numeric_data import matrix_from_dict, positive_budget
from ..quantum.states import validate_dimensions

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


def quantum_to_dict(strategy):
    return {"schema": QUANTUM_SCHEMA, "encoding": "complex_float64",
            "network": network_to_dict(strategy.network), "tolerance": float(strategy.tolerance),
            "sources": [{"dimensions": list(s.dimensions), "tolerance": float(s.tolerance),
                         "state": _matrix_to_dict(s.state)} for s in strategy.sources],
            "measurements": [[[_matrix_to_dict(effect) for effect in povm] for povm in party]
                             for party in strategy.measurements]}


def quantum_from_dict(data, *, max_total_dimension=1024, max_matrix_entries=1_000_000):
    """Check aggregate budgets before decoding/eigendecomposing any source matrix."""
    positive_budget(max_total_dimension, "max_total_dimension")
    positive_budget(max_matrix_entries, "max_matrix_entries")
    _fields(data, {"schema", "encoding", "network", "tolerance", "sources", "measurements"}, QUANTUM_SCHEMA)
    if data["encoding"] != "complex_float64":
        raise ValueError("Unsupported quantum network numerical encoding")
    if not isinstance(data["sources"], list) or not isinstance(data["measurements"], list):
        raise ValueError("Sources and measurements must be lists")
    network = network_from_dict(data["network"])
    if len(data["sources"]) != len(network.sources):
        raise ValueError("Provide one source per network source")
    dimensions = []
    for source, endpoints in zip(data["sources"], network.sources):
        _fields(source, {"dimensions", "tolerance", "state"})
        dims = validate_dimensions(source["dimensions"])
        if len(dims) != len(endpoints):
            raise ValueError("Source dimensions do not match endpoint factors")
        dimensions.append(dims)
    size = prod(prod(dims) for dims in dimensions)
    if size > max_total_dimension:
        raise ValueError("Product source dimension exceeds max_total_dimension")
    local_dims = tuple(prod(dimensions[s][e] for s, e in ports) for ports in network.party_ports)
    if len(data["measurements"]) != network.parties:
        raise ValueError("Provide one measurement collection per party")
    entries = size * size + sum(prod(dims)**2 for dims in dimensions)
    for party, dim in zip(data["measurements"], local_dims):
        if not isinstance(party, list) or not party or any(not isinstance(p, list) or not p for p in party):
            raise ValueError("Measurements must be nonempty nested party/input/outcome lists")
        entries += sum(len(p) * dim * dim for p in party)
    if entries > max_matrix_entries:
        raise ValueError("Network matrices exceed max_matrix_entries")
    sources = tuple(QuantumSource(matrix_from_dict(s["state"], prod(dims)), dims, s["tolerance"])
                    for s, dims in zip(data["sources"], dimensions))
    measurements = tuple(tuple(tuple(matrix_from_dict(e, dim) for e in povm) for povm in party)
                         for party, dim in zip(data["measurements"], local_dims))
    return QuantumNetworkStrategy(network, sources, measurements, data["tolerance"], max_total_dimension)
