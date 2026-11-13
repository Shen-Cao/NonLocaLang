"""Local response relabellings that retain explicit independent-source witnesses."""
from math import prod

import numpy as np

from ..core.numeric_data import positive_budget
from ..networks.classical import ClassicalNetworkStrategy
from ..networks.quantum import QuantumNetworkStrategy
from .maps import RelabellingBudgetExceeded
from .quantum import NumericalRelabellingError, _check_measurement_interface, _relabel_measurements
from .resources import _relabel_network, _require_map


def relabel_classical_network_strategy(strategy, relabelling, *, max_entries=1_000_000):
    """Reindex exact local conditional tables without introducing shared randomness.

    Source laws and source-index order stay unchanged. max_entries counts source
    probabilities and response-table entries together before new tables are built.
    """
    if not isinstance(strategy, ClassicalNetworkStrategy):
        raise TypeError('Expected ClassicalNetworkStrategy')
    _require_map(relabelling)
    if strategy.scenario != relabelling.source:
        raise ValueError('Network strategy interface differs from the relabelling source')
    limit = positive_budget(max_entries, 'max_entries')
    entries = sum(map(len, strategy.source_probabilities)) + sum(map(len, strategy.responses))
    if entries > limit:
        raise RelabellingBudgetExceeded('Classical network relabelling exceeds max_entries')
    network = _relabel_network(strategy.network, relabelling)
    responses = []
    for new, old in enumerate(relabelling.party_permutation):
        outputs = relabelling.source.output_sizes[old]
        latent_count = prod(len(strategy.source_probabilities[s])
                            for s in strategy.network.incident_sources(old))
        table = strategy.responses[old]
        responses.append(tuple(table[(old_input*latent_count + latent)*outputs + old_output]
                               for x, old_input in enumerate(relabelling.input_maps[new])
                               for latent in range(latent_count)
                               for old_output in relabelling.output_maps[new][x]))
    return ClassicalNetworkStrategy(network, relabelling.target, strategy.source_probabilities, tuple(responses))


def _quantum_counts(strategy, max_total_dimension, max_matrix_entries):
    total_limit = positive_budget(max_total_dimension, 'max_total_dimension')
    entry_limit = positive_budget(max_matrix_entries, 'max_matrix_entries')
    total = 1
    for dimension in strategy.dimensions:
        if dimension > total_limit // total:
            raise RelabellingBudgetExceeded('Quantum network relabelling exceeds max_total_dimension')
        total *= dimension
    entries = total*total + sum(source.state.size for source in strategy.sources)
    entries += sum(len(povm)*dimension**2 for dimension, party in zip(strategy.dimensions, strategy.measurements)
                   for povm in party)
    if entries > entry_limit:
        raise RelabellingBudgetExceeded('Quantum source, global-state, and measurement data exceed max_matrix_entries')
    return total_limit


def relabel_quantum_network_strategy(strategy, relabelling, *, max_total_dimension=1024,
                                     max_matrix_entries=1_000_000):
    """Rename source endpoints and reindex local POVMs, retaining every source state.

    Endpoint positions are not sorted: their order defines source tensor factors.
    Neither global nor source states are repaired, and no tolerance is widened.
    A failed floating revalidation remains an explicit numerical transport failure.
    """
    if not isinstance(strategy, QuantumNetworkStrategy):
        raise TypeError('Expected QuantumNetworkStrategy; source provenance must remain explicit')
    _require_map(relabelling)
    if strategy.network.parties != relabelling.source.parties:
        raise ValueError('Network strategy and relabelling have different party counts')
    _check_measurement_interface(strategy.measurements, relabelling)
    total_limit = _quantum_counts(strategy, max_total_dimension, max_matrix_entries)
    network = _relabel_network(strategy.network, relabelling)
    measurements = _relabel_measurements(strategy.measurements, relabelling)
    try:
        strategy.validate()
    except (ValueError, np.linalg.LinAlgError) as error:
        raise NumericalRelabellingError('Source quantum network failed revalidation at its unchanged tolerance') from error
    try:
        return QuantumNetworkStrategy(network, strategy.sources, measurements, strategy.tolerance,
                                      max_total_dimension=total_limit)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise NumericalRelabellingError(
            'Transported quantum network failed revalidation at the unchanged tolerance; no source repair was applied') from error
