"""Task integration that retains independent-source witness provenance."""
from dataclasses import replace

from ..analysis import AnalysisResult
from ..core.resources import AnalysisTask, Resource
from ..quantum.validation import validate_strategy
from .classical import ClassicalNetworkStrategy
from .quantum import QuantumNetworkStrategy


def evaluate(task, strategy):
    network = task.resource.network
    if network is None:
        raise ValueError("Expected a declared source network")
    expected = ClassicalNetworkStrategy if task.resource.kind == Resource.LOCAL else QuantumNetworkStrategy
    if not isinstance(strategy, expected):
        raise TypeError(f"This network task requires a {expected.__name__}")
    if strategy.network != network:
        raise ValueError("Witness source graph or factor ordering differs from the contract")
    if isinstance(strategy, ClassicalNetworkStrategy):
        return AnalysisResult(strategy.score(task.game), "classical_network", "network_evaluation",
                              "exact_witness", "evaluated", {"source_independence": "explicit_product",
                              "scope": "supplied witness score, not an optimized value"}, strategy)
    plain_task = AnalysisTask(task.game, replace(task.resource, network=None))
    diagnostics = validate_strategy(plain_task, strategy.as_strategy())
    diagnostics.update({"source_independence": "explicit_product", "sources": len(network.sources),
                        "scope": "supplied witness score, not an optimized value"})
    return AnalysisResult(strategy.score(task.game), "quantum_network", "network_evaluation",
                          "numerical_witness", "validated", diagnostics, strategy)


def deterministic_quantum_network_strategy(network, scenario, tables):
    """Realize local deterministic tables with one-dimensional independent sources."""
    import numpy as np
    from .classical import deterministic_network_strategy
    from .quantum import QuantumSource
    deterministic_network_strategy(network, scenario, tables)
    sources = tuple(QuantumSource(np.ones((1, 1)), (1,) * len(endpoints))
                    for endpoints in network.sources)
    measurements = tuple(tuple(tuple(np.array([[int(a == tables[i][x])]], dtype=complex)
                                      for a in range(scenario.output_sizes[i]))
                                for x in range(scenario.input_sizes[i]))
                         for i in range(scenario.parties))
    return QuantumNetworkStrategy(network, sources, measurements)
