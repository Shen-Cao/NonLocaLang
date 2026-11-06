"""State optimization for fixed measurements, optionally with PPT constraints.

PPT is an explicit state restriction; it is not identified with separability.
See Peres, https://arxiv.org/abs/quant-ph/9604005.
"""
from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np

from ._blocks import bell_operator
from .states import partial_transpose, state_diagnostics, validate_ppt_cuts
from .strategy import QuantumStrategy


@dataclass(frozen=True)
class StateOptimizationResult:
    """Numerical optimum of the fixed-measurement state block and a repaired witness.

    ``objective`` is the raw optimizer/eigenvalue output. ``value`` is evaluated
    on the returned strategy after repair. Neither is a certified bound on the
    full game value, and the measurement operators are never optimized here.
    """
    strategy: QuantumStrategy
    value: float
    objective: float
    status: str
    diagnostics: dict
    guarantee: str = "numerical_feasible"


def _repair_state(raw, dimensions, cuts, tolerance):
    raw = np.asarray(raw, dtype=complex)
    if not np.isfinite(raw).all():
        raise RuntimeError("State SDP returned nonfinite values")
    state = (raw + raw.conj().T) / 2
    trace = float(np.trace(state).real)
    if trace <= 0:
        raise RuntimeError("State SDP returned a nonpositive trace")
    state /= trace
    size = prod(dimensions)
    minimum = min([float(np.linalg.eigvalsh(state).min())] + [
        float(np.linalg.eigvalsh(partial_transpose(state, dimensions, cut)).min())
        for cut in cuts])
    margin = min(1 / (2 * size), max(10 * np.finfo(float).eps * size, tolerance * 0.01))
    mixing = max(0.0, (margin - minimum) / (1 / size - minimum)) if minimum < margin else 0.0
    state = (1 - mixing) * state + mixing * np.eye(size) / size
    diagnostics = state_diagnostics(state, dimensions, tolerance=tolerance, ppt_cuts=cuts)
    diagnostics.update({"identity_mixing_weight": mixing, "raw_trace_residual": abs(trace - 1),
                        "normalized_raw_minimum_constraint_eigenvalue": minimum})
    return state, diagnostics


def state_update(bell, dimensions, cuts, solver, options, tolerance):
    """Solve a single state block. Cuts are assumed to be canonical and validated."""
    if not cuts:
        values, vectors = np.linalg.eigh(bell)
        state = np.outer(vectors[:, -1], vectors[:, -1].conj())
        return state, {"objective": float(values[-1]), "status": "eigenvalue",
                       "solver": "numpy.linalg.eigh", "identity_mixing_weight": 0.0,
                       **state_diagnostics(state, dimensions, tolerance=tolerance)}
    import cvxpy as cp

    size = prod(dimensions)
    rho = cp.Variable((size, size), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]
    for cut in cuts:
        transposed = rho
        for party in cut:
            transposed = cp.partial_transpose(transposed, dimensions, axis=party)
        constraints.append(transposed >> 0)
    problem = cp.Problem(cp.Maximize(cp.real(cp.trace(rho @ bell))), constraints)
    problem.solve(solver=solver, **options)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or rho.value is None:
        raise RuntimeError(f"State SDP did not return a usable solution: {problem.status}")
    if problem.value is None or not np.isfinite(problem.value):
        raise RuntimeError("State SDP returned a nonfinite objective")
    state, diagnostics = _repair_state(rho.value, dimensions, cuts, tolerance)
    diagnostics.update({"objective": float(problem.value), "status": str(problem.status),
                        "solver": problem.solver_stats.solver_name,
                        "solver_iterations": problem.solver_stats.num_iters,
                        "max_raw_constraint_violation": float(max(
                            np.max(c.violation()) for c in constraints))})
    return state, diagnostics


def optimize_state(game, measurements, *, ppt_cuts=(), solver="SCS",
                   validation_tolerance=1e-7, solver_options=None, max_joint_dimension=256):
    """Optimize a density matrix while all supplied local POVMs remain fixed.

    Supports any finite party count. Without cuts the state block is an
    eigenvalue problem; with PPT cuts it is an SDP. Numerical state repair mixes
    with the maximally mixed state to enforce every positivity constraint
    simultaneously, then recomputes the actual score. This remains uncertified.
    """
    if (isinstance(max_joint_dimension, bool) or not isinstance(max_joint_dimension, Integral)
            or max_joint_dimension < 1):
        raise ValueError("max_joint_dimension must be a positive integer")
    if (isinstance(validation_tolerance, bool) or not np.isfinite(validation_tolerance)
            or validation_tolerance <= 0):
        raise ValueError("validation_tolerance must be positive and finite")
    measurements = tuple(tuple(tuple(povm) for povm in party) for party in measurements)
    if not measurements or any(not p or any(not povm for povm in p) for p in measurements):
        raise ValueError("Measurements need at least one party, setting, and outcome")
    try:
        dimensions = tuple(np.asarray(p[0][0]).shape[0] for p in measurements)
    except IndexError as error:
        raise ValueError("Measurement effects must be matrices") from error
    if prod(dimensions) > max_joint_dimension:
        raise ValueError("State optimization exceeds max_joint_dimension")
    if not prod(dimensions):
        raise ValueError("Measurement dimensions must be positive")
    reference = QuantumStrategy(np.eye(prod(dimensions)) / prod(dimensions), measurements,
                                validation_tolerance)
    reference.probabilities(game.scenario)
    cuts = validate_ppt_cuts(ppt_cuts, len(dimensions))
    options = dict(solver_options or {})
    if "solver" in options:
        raise ValueError("Pass the solver through its explicit argument")
    bell = bell_operator(game, reference.measurements, dimensions)
    state, diagnostics = state_update(bell, dimensions, cuts, solver, options, validation_tolerance)
    strategy = QuantumStrategy(state, reference.measurements, validation_tolerance)
    value = strategy.score(game)
    diagnostics.update({"ppt_cuts": cuts, "local_dimensions": dimensions, "certified": False,
                        "repair_score_change": value - diagnostics["objective"],
                        "objective_guarantee": "numerical_block_optimum"})
    return StateOptimizationResult(strategy, value, diagnostics["objective"],
                                   diagnostics["status"], diagnostics)
