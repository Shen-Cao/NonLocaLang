"""Exact point evaluation under the resource restrictions actually checked here."""
from ...analysis.result import AnalysisResult
from ...core.identity import resource_id
from ...core.resources import MeasurementClass, Resource, StateConstraint
from .strategy import ExactQuantumStrategy


def evaluate(task, strategy):
    resource = task.resource
    if not isinstance(strategy, ExactQuantumStrategy):
        raise TypeError('Expected ExactQuantumStrategy')
    if (resource.kind != Resource.QUANTUM_FINITE or resource.network is not None
            or resource.fixed_state is not None or resource.state_constraint != StateConstraint.ANY
            or resource.measurement_class != MeasurementClass.POVM):
        raise NotImplementedError('Exact point evaluation currently checks finite tensor POVMs and dimension bounds only')
    if len(strategy.dimensions) != task.game.scenario.parties:
        raise ValueError('Exact strategy and task have different parties')
    if resource.local_dimensions is not None and any(
            actual > limit for actual, limit in zip(strategy.dimensions, resource.local_dimensions)):
        raise ValueError('Exact strategy exceeds declared local dimension bounds')
    diagnostics = strategy.validate()
    value = strategy.score(task.game)
    return AnalysisResult(value, resource.kind.value, 'exact_quantum_point_evaluation',
                          'exact_witness', 'evaluated', {**diagnostics,
                          'resource_id': resource_id(resource),
                          'scope': 'exact supplied strategy score; no optimum claim'}, strategy)
