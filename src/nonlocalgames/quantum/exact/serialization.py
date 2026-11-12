"""Strict exact strategy data; no tolerance or numerical encoding is accepted."""
from .budgets import ExactBudgetExceeded, bounded_dimension, coerce_budget
from .matrix import RationalMatrix
from .scalars import scalar_string
from .strategy import ExactQuantumStrategy

STRATEGY_SCHEMA = 'nonlocalgames.exact_quantum_strategy.v1'


def matrix_to_dict(matrix, *, budget=None):
    if not isinstance(matrix, RationalMatrix):
        raise TypeError('Expected RationalMatrix')
    matrix.check_budget(budget)
    return {'real': [[scalar_string(v) for v in row] for row in matrix.real],
            'imag': [[scalar_string(v) for v in row] for row in matrix.imag]}


def matrix_from_dict(data, dimension=None, *, budget=None):
    budget = coerce_budget(budget)
    if not isinstance(data, dict) or set(data) != {'real', 'imag'}:
        raise ValueError('Exact matrix data require real and imag fields only')
    if not isinstance(data['real'], list):
        raise ValueError('Exact matrix rows must be JSON lists')
    dimension = bounded_dimension(len(data['real']) if dimension is None else dimension, budget)
    if dimension**2 > budget.max_matrix_entries:
        raise ExactBudgetExceeded('Exact matrix exceeds max_matrix_entries')
    for part in (data['real'], data['imag']):
        if not isinstance(part, list) or len(part) != dimension or any(
            not isinstance(row, list) or len(row) != dimension for row in part
        ):
            raise ValueError('Exact matrix shape differs from its declared dimension')
        if any(not isinstance(v, str) for row in part for v in row):
            raise ValueError('Exact wire entries must be canonical rational strings')
    return RationalMatrix(data['real'], data['imag'], budget=budget)


def strategy_to_dict(strategy, *, budget=None):
    if not isinstance(strategy, ExactQuantumStrategy):
        raise TypeError('Expected ExactQuantumStrategy')
    budget = strategy._budget if budget is None else coerce_budget(budget)
    strategy.validate(budget=budget)
    return {'schema': STRATEGY_SCHEMA, 'encoding': 'rational_complex',
            'dimensions': list(strategy.dimensions), 'state': matrix_to_dict(strategy.state, budget=budget),
            'measurements': [[[matrix_to_dict(effect, budget=budget) for effect in povm]
                              for povm in party] for party in strategy.measurements]}


def strategy_from_dict(data, *, budget=None):
    """Preflight dimensions/counts, then exactly validate all state and POVM data."""
    budget = coerce_budget(budget)
    fields = {'schema', 'encoding', 'dimensions', 'state', 'measurements'}
    if not isinstance(data, dict) or set(data) != fields or data['schema'] != STRATEGY_SCHEMA:
        raise ValueError('Invalid exact quantum strategy fields or schema')
    if data['encoding'] != 'rational_complex':
        raise ValueError('Exact quantum strategies require rational_complex encoding')
    raw_dimensions = data['dimensions']
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ValueError('Exact strategy dimensions must be a nonempty JSON list')
    if len(raw_dimensions) > budget.max_parties:
        raise ExactBudgetExceeded('Exact strategy exceeds max_parties')
    dimensions, total = [], 1
    for value in raw_dimensions:
        dimension = bounded_dimension(value, budget)
        total *= dimension
        if total > budget.max_joint_dimension:
            raise ExactBudgetExceeded('Exact strategy exceeds max_joint_dimension')
        dimensions.append(dimension)
    if 2*total > budget.max_psd_size:
        raise ExactBudgetExceeded('Exact strategy realification exceeds max_psd_size')
    entries = total**2
    if entries > budget.max_matrix_entries:
        raise ExactBudgetExceeded('Exact state exceeds max_matrix_entries')
    measurements = data['measurements']
    if not isinstance(measurements, list) or len(measurements) != len(dimensions):
        raise ValueError('Provide one measurement collection per local tensor factor')
    for dimension, party in zip(dimensions, measurements):
        if not isinstance(party, list) or not party:
            raise ValueError('Each party needs a nonempty list of POVMs')
        for povm in party:
            if not isinstance(povm, list) or not povm:
                raise ValueError('Each POVM needs a nonempty list of effects')
            entries += len(povm)*dimension**2
            if entries > budget.max_matrix_entries:
                raise ExactBudgetExceeded('Exact state and POVMs exceed max_matrix_entries')
    state = matrix_from_dict(data['state'], total, budget=budget)
    exact_measurements = tuple(tuple(tuple(matrix_from_dict(effect, dimension, budget=budget)
                                          for effect in povm) for povm in party)
                               for dimension, party in zip(dimensions, measurements))
    return ExactQuantumStrategy(state, exact_measurements, tuple(dimensions), budget=budget)
