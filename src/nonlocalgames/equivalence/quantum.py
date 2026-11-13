"""Quantum witness transport by local interface and party-factor relabelling."""
from numbers import Integral
from operator import index

from ..core.numeric_data import positive_budget
from .maps import RelabellingBudgetExceeded


class NumericalRelabellingError(ValueError):
    """Unchanged numerical data/tolerance did not pass revalidation after transport."""


def _factor_preflight(dimensions, permutation, max_joint_dimension, max_matrix_entries):
    joint_limit = positive_budget(max_joint_dimension, 'max_joint_dimension')
    entry_limit = positive_budget(max_matrix_entries, 'max_matrix_entries')
    if not isinstance(dimensions, (tuple, list)) or not dimensions:
        raise TypeError('Tensor dimensions must be a nonempty finite sequence')
    if any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in dimensions):
        raise ValueError('Tensor dimensions must be positive integers')
    dimensions = tuple(index(d) for d in dimensions)
    if not isinstance(permutation, (tuple, list)) or len(permutation) != len(dimensions):
        raise ValueError('Party permutation must match the tensor factors')
    if any(isinstance(p, bool) or not isinstance(p, Integral) for p in permutation):
        raise ValueError('Party permutation entries must be integers')
    permutation = tuple(index(p) for p in permutation)
    if sorted(permutation) != list(range(len(dimensions))):
        raise ValueError('Party permutation must be bijective')
    total = 1
    for dimension in dimensions:
        if dimension > joint_limit // total:
            raise RelabellingBudgetExceeded('Quantum relabelling exceeds max_joint_dimension')
        total *= dimension
    if total > entry_limit // total:
        raise RelabellingBudgetExceeded('Quantum state relabelling exceeds max_matrix_entries')
    return dimensions, permutation, total


def _basis_indices(dimensions, permutation, total):
    """New basis index -> old basis index; no physical dimension is changed."""
    target_dimensions = tuple(dimensions[p] for p in permutation)
    indices = []
    for new_index in range(total):
        remainder, source_coordinates = new_index, [0]*len(dimensions)
        for target_party in range(len(dimensions)-1, -1, -1):
            remainder, coordinate = divmod(remainder, target_dimensions[target_party])
            source_coordinates[permutation[target_party]] = coordinate
        source_index = 0
        for coordinate, dimension in zip(source_coordinates, dimensions):
            source_index = source_index*dimension + coordinate
        indices.append(source_index)
    return tuple(indices)


def _permute_numeric_density(state, dimensions, party_permutation, *,
                             max_joint_dimension=1024, max_matrix_entries=1_000_000):
    """Pure index selection, preserving complex components and signed zeros.

    A basis-index permutation avoids an artificial NumPy tensor-axis limit for
    many one-dimensional parties. This helper performs no physical validation.
    """
    dimensions, permutation, total = _factor_preflight(
        dimensions, party_permutation, max_joint_dimension, max_matrix_entries)
    import numpy as np
    if not isinstance(state, np.ndarray) or state.shape != (total, total):
        raise ValueError('Density matrix does not match its declared tensor factors')
    indices = _basis_indices(dimensions, permutation, total)
    return state[np.ix_(indices, indices)]


def _check_measurement_interface(measurements, relabelling):
    source = relabelling.source
    if len(measurements) != source.parties or any(
        len(party) != nx or any(len(povm) != na for povm in party)
        for party, nx, na in zip(measurements, source.input_sizes, source.output_sizes)
    ):
        raise ValueError('Strategy measurements do not match the relabelling source interface')


def _relabel_measurements(measurements, relabelling):
    """Select existing effects; output maps are indexed by each NEW input."""
    _check_measurement_interface(measurements, relabelling)
    return tuple(tuple(tuple(measurements[old_party][relabelling.input_maps[new_party][x]][a]
                             for a in relabelling.output_maps[new_party][x])
                       for x in range(relabelling.target.input_sizes[new_party]))
                 for new_party, old_party in enumerate(relabelling.party_permutation))


def _require_map(relabelling):
    from .maps import Relabelling
    if not isinstance(relabelling, Relabelling):
        raise TypeError('Expected a checked Relabelling descriptor')


def relabel_quantum_strategy(strategy, relabelling, *, max_joint_dimension=1024,
                              max_matrix_entries=1_000_000):
    """Transport a numerical tensor witness without repair or tolerance widening.

    This is not a source-network witness transport. Numerical revalidation may
    fail at a floating tolerance boundary even for an exact index permutation.
    """
    from ..quantum.strategy import QuantumStrategy
    if not isinstance(strategy, QuantumStrategy):
        raise TypeError('Expected a plain QuantumStrategy; network provenance requires network transport')
    _require_map(relabelling)
    dimensions, _, total = _factor_preflight(strategy.dimensions, relabelling.party_permutation,
                                             max_joint_dimension, max_matrix_entries)
    if len(dimensions) != relabelling.source.parties:
        raise ValueError('Strategy parties do not match the relabelling source')
    entries = total**2 + sum(len(povm)*dimension**2 for dimension, party in
                             zip(dimensions, strategy.measurements) for povm in party)
    if entries > positive_budget(max_matrix_entries, 'max_matrix_entries'):
        raise RelabellingBudgetExceeded('State and measurements exceed max_matrix_entries')
    measurements = _relabel_measurements(strategy.measurements, relabelling)
    import numpy as np
    try:
        strategy.validate()
    except (ValueError, np.linalg.LinAlgError) as error:
        raise NumericalRelabellingError('Source numerical witness failed revalidation at its unchanged tolerance') from error
    state = _permute_numeric_density(strategy.state, dimensions, relabelling.party_permutation,
                                     max_joint_dimension=max_joint_dimension,
                                     max_matrix_entries=max_matrix_entries)
    try:
        return QuantumStrategy(state, measurements, strategy.tolerance)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise NumericalRelabellingError(
            'Transported numerical witness failed revalidation at the unchanged tolerance; no repair was applied') from error


def relabel_exact_strategy(strategy, relabelling, *, budget=None):
    """Transport exact matrices in original dimensions and revalidate the result.

    The witness's exact budget is inherited unless explicitly overridden. Budget
    exhaustion does not disprove the mathematical validity of the relabelling.
    """
    from ..quantum.exact import ExactQuantumStrategy, RationalMatrix
    from ..quantum.exact.budgets import coerce_budget
    if not isinstance(strategy, ExactQuantumStrategy):
        raise TypeError('Expected ExactQuantumStrategy')
    _require_map(relabelling)
    policy = strategy._budget if budget is None else coerce_budget(budget)
    _check_measurement_interface(strategy.measurements, relabelling)
    strategy.validate(budget=policy)
    measurements = _relabel_measurements(strategy.measurements, relabelling)
    total = strategy.state.dimension
    indices = _basis_indices(strategy.dimensions, relabelling.party_permutation, total)
    state = RationalMatrix(tuple(tuple(strategy.state.real[i][j] for j in indices) for i in indices),
                           tuple(tuple(strategy.state.imag[i][j] for j in indices) for i in indices), budget=policy)
    dimensions = tuple(strategy.dimensions[p] for p in relabelling.party_permutation)
    return ExactQuantumStrategy(state, measurements, dimensions, budget=policy)
