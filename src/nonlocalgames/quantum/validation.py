"""Check a concrete quantum strategy against a composable resource contract."""
from math import prod
from numbers import Integral

import numpy as np

from ..core.resources import MeasurementClass, Resource, StateConstraint
from .states import product_state_residual, state_diagnostics
from .strategy import QuantumStrategy


def validate_strategy(task, strategy):
    """Validate one branch; external shared mixing does not relax branch constraints."""
    r = task.resource
    if r.kind != Resource.QUANTUM_FINITE:
        raise NotImplementedError("Explicit tensor-strategy checking requires quantum_finite")
    if not isinstance(strategy, QuantumStrategy):
        raise TypeError("Expected QuantumStrategy")
    diagnostics = strategy.validate()
    s = task.game.scenario
    if len(strategy.dimensions) != s.parties or any(
        len(party) != nx or any(len(povm) != na for povm in party)
        for party, nx, na in zip(strategy.measurements, s.input_sizes, s.output_sizes)
    ):
        raise ValueError("Strategy and game interfaces differ")
    if r.local_dimensions is not None and any(a > b for a, b in zip(strategy.dimensions, r.local_dimensions)):
        raise ValueError("Strategy exceeds declared local dimension bounds")
    tolerance = strategy.tolerance
    if r.fixed_state is not None:
        if strategy.dimensions != r.fixed_state.dimensions:
            raise ValueError("Strategy uses a different fixed-state tensor factorization")
        residual = float(np.linalg.norm(strategy.state - r.fixed_state.state, ord=2))
        if residual > r.fixed_state.tolerance:
            raise ValueError("Strategy does not use the declared fixed state")
        diagnostics["fixed_state_residual"] = residual
        diagnostics["fixed_state_tolerance"] = r.fixed_state.tolerance
    diagnostics.update(state_diagnostics(strategy.state, strategy.dimensions,
                                         tolerance=tolerance, ppt_cuts=r.ppt_cuts))
    if r.state_constraint == StateConstraint.PRODUCT:
        residual = product_state_residual(strategy.state, strategy.dimensions)
        if residual > tolerance:
            raise ValueError("Strategy state is not product within tolerance")
        diagnostics["product_state_residual"] = residual
    if r.measurement_class == MeasurementClass.PROJECTIVE:
        residual = max(float(np.linalg.norm(e @ f - (e if i == j else 0), ord=2))
                       for party in strategy.measurements for povm in party
                       for i, e in enumerate(povm) for j, f in enumerate(povm))
        if residual > tolerance:
            raise ValueError("Strategy measurements are not projective within tolerance")
        diagnostics["projectivity_residual"] = residual
    diagnostics.update({"validation_tolerance": tolerance, "certified": False})
    return diagnostics


def evaluate(task, strategy):
    from ..analysis import AnalysisResult
    diagnostics = validate_strategy(task, strategy)
    return AnalysisResult(strategy.score(task.game), task.resource.kind.value,
                          "restricted_strategy_evaluation", "numerical_witness", "validated",
                          diagnostics, strategy)


def validate_measurements(task, measurements, *, tolerance=1e-7, max_joint_dimension=256):
    """Fail before optimization if supplied POVMs violate interface or resource bounds."""
    if (isinstance(max_joint_dimension, bool) or not isinstance(max_joint_dimension, Integral)
            or max_joint_dimension < 1):
        raise ValueError("max_joint_dimension must be a positive integer")
    measurements = tuple(tuple(tuple(povm) for povm in party) for party in measurements)
    if not measurements or any(not p or any(not povm for povm in p) for p in measurements):
        raise ValueError("Measurements need nonempty parties, settings, and outcomes")
    try:
        dimensions = tuple(np.asarray(p[0][0]).shape[0] for p in measurements)
    except IndexError as error:
        raise ValueError("Measurement effects must be matrices") from error
    size = prod(dimensions)
    if size < 1 or size > max_joint_dimension:
        raise ValueError("Measurement dimension is zero or exceeds max_joint_dimension")
    state = np.eye(size) / size
    reference = QuantumStrategy(state, measurements, tolerance)
    validate_strategy(task, reference)
    return reference.measurements
