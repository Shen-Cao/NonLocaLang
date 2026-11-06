"""Alternating finite-dimensional quantum strategy search.

This heuristic searches bipartite tensor-product strategies. Each state update
uses a largest eigenvector and each local measurement update uses an SDP.
Neither convergence nor a numerical feasible witness certifies global optimality.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from .strategy import QuantumStrategy

if TYPE_CHECKING:
    from nonlocalgames import Game


@dataclass(frozen=True)
class SeesawResult:
    """Best observed witness and its unmodified sweep history.

    ``history`` records the initial score and completed sweeps of the restart
    containing the best witness. ``iterations`` locates that witness in the
    history. Later entries can be smaller due to numerical solver error.
    """

    strategy: QuantumStrategy
    value: float
    history: tuple[float, ...]
    restart_scores: tuple[float, ...]
    iterations: int
    seed: int
    solver: str
    solver_statuses: tuple[str, ...]


def _normalize_povm(effects: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """PSD projection followed by congruence normalization, without score repair."""
    projected = []
    for effect in effects:
        if not np.isfinite(effect).all():
            raise RuntimeError("The measurement SDP returned nonfinite effects.")
        values, vectors = np.linalg.eigh((effect + effect.conj().T) / 2)
        projected.append((vectors * np.maximum(values, 0)) @ vectors.conj().T)
    values, vectors = np.linalg.eigh(sum(projected))
    if values.min() <= np.finfo(float).eps:
        raise RuntimeError("The measurement effects cannot be normalized stably.")
    inverse_root = (vectors * (1 / np.sqrt(values))) @ vectors.conj().T
    return tuple(inverse_root @ effect @ inverse_root for effect in projected)


def _best_state(coefficients: np.ndarray, measurements: tuple, dimensions: tuple) -> np.ndarray:
    bell = np.zeros((np.prod(dimensions),) * 2, dtype=complex)
    for x, y, a, b in np.ndindex(coefficients.shape):
        bell += coefficients[x, y, a, b] * np.kron(measurements[0][x][a], measurements[1][y][b])
    _, vectors = np.linalg.eigh((bell + bell.conj().T) / 2)
    vector = vectors[:, -1]
    return np.outer(vector, vector.conj())


def _optimize_party(coefficients, state, measurements, dimensions, party, solver, options):
    import cvxpy as cp

    rho = state.reshape(dimensions[0], dimensions[1], dimensions[0], dimensions[1])
    updated, statuses = [], []
    for setting, old_povm in enumerate(measurements[party]):
        effective = [np.zeros((dimensions[party],) * 2, dtype=complex) for _ in old_povm]
        for x, y, a, b in np.ndindex(coefficients.shape):
            if (x, y)[party] != setting:
                continue
            if party == 0:
                marginal = np.einsum("ijkl,lj->ik", rho, measurements[1][y][b])
            else:
                marginal = np.einsum("ijkl,ki->jl", rho, measurements[0][x][a])
            effective[(a, b)[party]] += coefficients[x, y, a, b] * marginal
        effects = [cp.Variable((1, 1), symmetric=True) if dimensions[party] == 1
                   else cp.Variable((dimensions[party],) * 2, hermitian=True) for _ in old_povm]
        effective = [h.real if dimensions[party] == 1 else (h + h.conj().T) / 2 for h in effective]
        terms = [cp.sum(cp.multiply(effect, h.T)) for effect, h in zip(effects, effective)]
        objective = sum(cp.real(term) if term.is_complex() else term for term in terms)
        problem = cp.Problem(cp.Maximize(objective), [effect >> 0 for effect in effects] + [sum(effects) == np.eye(dimensions[party])])
        problem.solve(solver=solver, **options)
        statuses.append(str(problem.status))
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or any(effect.value is None for effect in effects):
            raise RuntimeError(f"Measurement SDP did not return a usable solution: {problem.status}")
        updated.append(_normalize_povm([effect.value for effect in effects]))
    return tuple(updated), statuses


def seesaw(
    game: Game,
    dimensions: tuple[int, int] = (2, 2),
    *,
    restarts: int = 3,
    max_iterations: int = 30,
    seed: int = 0,
    solver: str = "SCS",
    convergence_tolerance: float = 1e-8,
    validation_tolerance: float = 1e-7,
    solver_options: dict | None = None,
) -> SeesawResult:
    """Search local optima with seeded random POVMs and alternating exact blocks.

    SDP effects are projected to PSD and jointly normalized, then the complete
    strategy is checked by ``QuantumStrategy``. Scores are always recomputed
    from that resulting strategy. No floating result is a certified lower bound.
    The random seed reproduces initialization; solver/library variations can
    still change the result. Resource use grows with dense alphabet products.
    """
    if len(game.scenario.input_sizes) != 2:
        raise ValueError("The see-saw backend supports exactly two parties.")
    dimensions = tuple(dimensions)
    if len(dimensions) != 2 or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in dimensions):
        raise ValueError("dimensions must contain two positive integers.")
    for name, value in (("restarts", restarts), ("max_iterations", max_iterations)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    for name, value in (("convergence_tolerance", convergence_tolerance), ("validation_tolerance", validation_tolerance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    options = dict(solver_options or {})
    if "solver" in options:
        raise ValueError("Pass the solver through its explicit argument.")
    shape = (*game.scenario.input_sizes, *game.scenario.output_sizes)
    coefficients = np.array([float(game.coefficient(x, a)) for x in game.scenario.inputs for a in game.scenario.outputs]).reshape(shape)
    rng = np.random.default_rng(seed)
    best_strategy, best_value, best_history, best_iteration = None, -np.inf, (), 0
    restart_scores, statuses = [], []
    for _ in range(restarts):
        measurements = []
        for dimension, inputs, outputs in zip(dimensions, game.scenario.input_sizes, game.scenario.output_sizes):
            party = []
            for _ in range(inputs):
                gaussian = rng.normal(size=(outputs, dimension, dimension)) + 1j * rng.normal(size=(outputs, dimension, dimension))
                party.append(_normalize_povm([g @ g.conj().T for g in gaussian]))
            measurements.append(tuple(party))
        measurements = tuple(measurements)
        state = _best_state(coefficients, measurements, dimensions)
        strategy = QuantumStrategy(state, measurements, validation_tolerance)
        history = [strategy.score(game)]
        run_best, run_iteration, run_strategy = history[0], 0, strategy
        for iteration in range(1, max_iterations + 1):
            for party in (0, 1):
                updated, local_statuses = _optimize_party(coefficients, state, measurements, dimensions, party, solver, options)
                statuses.extend(local_statuses)
                measurements = (updated, measurements[1]) if party == 0 else (measurements[0], updated)
            state = _best_state(coefficients, measurements, dimensions)
            strategy = QuantumStrategy(state, measurements, validation_tolerance)
            score = strategy.score(game)
            history.append(score)
            if score > run_best:
                run_best, run_iteration, run_strategy = score, iteration, strategy
            if abs(history[-1] - history[-2]) <= convergence_tolerance:
                break
        restart_scores.append(run_best)
        if run_best > best_value:
            best_strategy, best_value, best_history, best_iteration = run_strategy, run_best, tuple(history), run_iteration
    return SeesawResult(best_strategy, best_value, best_history, tuple(restart_scores), best_iteration, int(seed), solver, tuple(statuses))
