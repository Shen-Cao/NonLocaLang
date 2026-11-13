"""Transport resource contracts together with their party and interface labels."""
from dataclasses import replace

import numpy as np

from ..core.numeric_data import positive_budget
from ..core.resources import AnalysisTask, ResourceSpec
from ..networks.structure import SourceNetwork
from ..quantum.states import FixedState, validate_ppt_cuts
from .maps import Relabelling
from .quantum import (NumericalRelabellingError, _factor_preflight,
                      _permute_numeric_density)


def _require_map(relabelling):
    if not isinstance(relabelling, Relabelling):
        raise TypeError('Expected a checked Relabelling descriptor')


def _inverse_parties(relabelling):
    inverse = [0]*relabelling.source.parties
    for new, old in enumerate(relabelling.party_permutation):
        inverse[old] = new
    return tuple(inverse)


def _relabel_network(network, relabelling):
    """Rename endpoints, retaining source order and each source's factor order."""
    if not isinstance(network, SourceNetwork):
        raise TypeError('Expected SourceNetwork')
    if network.parties != relabelling.source.parties:
        raise ValueError('Source graph and relabelling have different parties')
    inverse = _inverse_parties(relabelling)
    return SourceNetwork(network.parties, tuple(tuple(inverse[party] for party in source)
                                               for source in network.sources))


def _source_contract(spec, relabelling):
    count = relabelling.source.parties
    if spec.local_dimensions is not None and len(spec.local_dimensions) != count:
        raise ValueError('Resource dimension bounds differ from the source party count')
    if spec.fixed_state is not None and len(spec.fixed_state.dimensions) != count:
        raise ValueError('Fixed-state factors differ from the source party count')
    if spec.network is not None and spec.network.parties != count:
        raise ValueError('Source graph differs from the source party count')
    if spec.ppt_cuts:
        validate_ppt_cuts(spec.ppt_cuts, count)


def relabel_resource(spec, relabelling, *, max_joint_dimension=1024,
                     max_matrix_entries=1_000_000):
    """Move all party-sensitive constraints without repairing numerical data.

    Symbolic dimension upper bounds are only permuted; matrix budgets apply to
    actual fixed-state data. Numerical revalidation can fail at the unchanged
    tolerance after a basis permutation, and that failure is reported explicitly.
    PPT cuts retain their spelling unless ResourceSpec's existing canonicalization
    applies; no new quotient of resource identities is introduced here.
    """
    if not isinstance(spec, ResourceSpec):
        raise TypeError('Expected ResourceSpec')
    _require_map(relabelling)
    positive_budget(max_joint_dimension, 'max_joint_dimension')
    positive_budget(max_matrix_entries, 'max_matrix_entries')
    _source_contract(spec, relabelling)
    permutation = relabelling.party_permutation
    inverse = _inverse_parties(relabelling)
    dimensions = None if spec.local_dimensions is None else tuple(spec.local_dimensions[i] for i in permutation)
    cuts = tuple(tuple(inverse[party] for party in cut) for cut in spec.ppt_cuts)
    network = None if spec.network is None else _relabel_network(spec.network, relabelling)
    fixed = spec.fixed_state
    if fixed is not None:
        _factor_preflight(fixed.dimensions, permutation, max_joint_dimension, max_matrix_entries)
        try:
            fixed.validate(ppt_cuts=spec.ppt_cuts)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise NumericalRelabellingError('Source fixed state failed revalidation at its unchanged tolerance') from error
        state = _permute_numeric_density(fixed.state, fixed.dimensions, permutation,
            max_joint_dimension=max_joint_dimension, max_matrix_entries=max_matrix_entries)
        try:
            fixed = FixedState(state, tuple(fixed.dimensions[i] for i in permutation), fixed.tolerance)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise NumericalRelabellingError(
                'Transported fixed state failed revalidation at the unchanged tolerance; no repair was applied') from error
    try:
        return replace(spec, local_dimensions=dimensions, ppt_cuts=cuts, network=network, fixed_state=fixed)
    except (ValueError, np.linalg.LinAlgError) as error:
        if fixed is None:
            raise
        raise NumericalRelabellingError(
            'Transported fixed-state constraints failed revalidation at the unchanged tolerance; no repair was applied') from error


def relabel_task(task, relabelling, *, max_entries=1_000_000,
                 max_joint_dimension=1024, max_matrix_entries=1_000_000):
    """Transport a value or network-compatibility task, preserving its actual scope.

    A backend specialized to a fixed party labelling may no longer apply after
    transport. This function does not silently rename the task again to enable it.
    """
    from ..core.compatibility_tasks import NetworkCompatibilityTask
    from .finite import _matching, _table_budget, relabel_behavior, relabel_game
    if not isinstance(task, (AnalysisTask, NetworkCompatibilityTask)):
        raise TypeError('Expected AnalysisTask or NetworkCompatibilityTask')
    source = task.game.scenario if isinstance(task, AnalysisTask) else task.behavior.scenario
    _matching(source, relabelling)
    _table_budget(source, max_entries)
    resource = relabel_resource(task.resource, relabelling, max_joint_dimension=max_joint_dimension,
                                max_matrix_entries=max_matrix_entries)
    if isinstance(task, AnalysisTask):
        return AnalysisTask(relabel_game(task.game, relabelling, max_entries=max_entries), resource)
    return NetworkCompatibilityTask(relabel_behavior(task.behavior, relabelling, max_entries=max_entries), resource)
