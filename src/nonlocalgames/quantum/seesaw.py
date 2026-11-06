"""Multipartite finite-dimensional strategy search with explicit resource restrictions."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from ._blocks import bell_operator, normalize_povm, optimize_party
from .restricted import state_update
from ._measurements import random_projective, validate_measurement_class
from .states import FixedState, state_diagnostics, validate_dimensions, validate_ppt_cuts
from .strategy import QuantumStrategy
from ..core.numeric_data import positive_tolerance

if TYPE_CHECKING:
    from nonlocalgames import Game


@dataclass(frozen=True)
class SeesawResult:
    """Best numerical witness and unmodified winning-restart score history.

    ``iterations`` locates the best witness in ``history``. All restart best
    scores, SDP statuses, and state-update diagnostics remain available. None of
    these numerical records certifies global optimality or rigorous feasibility.
    """
    strategy: QuantumStrategy
    value: float
    history: tuple[float, ...]
    restart_scores: tuple[float, ...]
    iterations: int
    seed: int
    solver: str
    solver_statuses: tuple[str, ...]
    fixed_state: bool = False
    ppt_cuts: tuple[tuple[int, ...], ...] = ()
    state_updates: tuple[dict, ...] = ()
    guarantee: str = "numerical_feasible"
    measurement_class: str = "povm"


def seesaw(
    game: Game,
    dimensions: tuple[int, ...] | None = None,
    *,
    fixed_state: FixedState | None = None,
    measurement_class: str = "povm",
    ppt_cuts=(),
    restarts: int = 3,
    max_iterations: int = 30,
    seed: int = 0,
    solver: str = "SCS",
    convergence_tolerance: float = 1e-8,
    validation_tolerance: float = 1e-7,
    solver_options: dict | None = None,
    max_joint_dimension: int = 256,
    max_update_terms: int = 2_000_000,
    max_measurement_entries: int = 1_000_000,
) -> SeesawResult:
    """Alternate local POVM updates and optional constrained state updates.

    With ``fixed_state``, only the POVMs change. Otherwise state blocks use a
    largest eigenvector or, when ``ppt_cuts`` is supplied, a PPT-constrained SDP.
    Dimensions default to the fixed state's dimensions, or qubits otherwise.
    Projective searches support one or two outcomes via analytic local spectral
    updates in the declared dimensions, including zero/full projectors.
    ``max_update_terms`` bounds event terms per full measurement/state sweep;
    it is an operation-count safeguard, not a runtime or memory prediction.
    ``max_measurement_entries`` separately bounds sum_i |X_i||A_i| d_i^2
    before constructing POVMs or local effective-objective matrices.
    Numerical repairs preserve normalization and declared PPT cuts; returned
    scores are recomputed after repair and are never clipped. This local search
    has no global-optimality guarantee. The seed controls initialization only.
    """
    parties = game.scenario.parties
    measurement_class = validate_measurement_class(measurement_class, game.scenario.output_sizes)
    if fixed_state is not None and not isinstance(fixed_state, FixedState):
        raise TypeError("fixed_state must be a FixedState specification")
    if dimensions is None:
        dimensions = fixed_state.dimensions if fixed_state is not None else (2,) * parties
    dimensions = validate_dimensions(dimensions)
    if len(dimensions) != parties:
        raise ValueError("dimensions must contain one positive integer per party")
    for name, value in (("restarts", restarts), ("max_iterations", max_iterations),
                        ("max_joint_dimension", max_joint_dimension), ("max_update_terms", max_update_terms),
                        ("max_measurement_entries", max_measurement_entries)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (parties + 1) * game.scenario.entries > max_update_terms:
        raise ValueError("Full search sweep exceeds max_update_terms")
    measurement_entries = sum(nx*na*d*d for nx, na, d in zip(
        game.scenario.input_sizes, game.scenario.output_sizes, dimensions))
    if measurement_entries > max_measurement_entries:
        raise ValueError("Measurement initialization exceeds max_measurement_entries")
    if prod(dimensions) > max_joint_dimension:
        raise ValueError("State search exceeds max_joint_dimension")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    convergence_tolerance = positive_tolerance(convergence_tolerance)
    validation_tolerance = positive_tolerance(validation_tolerance)
    cuts = validate_ppt_cuts(ppt_cuts, parties)
    if fixed_state is not None:
        if fixed_state.dimensions != dimensions:
            raise ValueError("Search dimensions must equal the supplied fixed state's dimensions")
        state_diagnostics(fixed_state.state, dimensions, tolerance=validation_tolerance,
                          ppt_cuts=cuts)
    options = dict(solver_options or {})
    if "solver" in options:
        raise ValueError("Pass the solver through its explicit argument")
    shape = (*game.scenario.input_sizes, *game.scenario.output_sizes)
    coefficients = np.array([float(game.coefficient(x, a)) for x in game.scenario.inputs
                             for a in game.scenario.outputs]).reshape(shape)
    rng = np.random.default_rng(seed)
    best_strategy, best_value, best_history, best_iteration = None, -np.inf, (), 0
    restart_scores, statuses, state_updates = [], [], []

    def update_state(measurements):
        if fixed_state is not None:
            return fixed_state.state
        bell = bell_operator(game, measurements, dimensions)
        state, diagnostics = state_update(bell, dimensions, cuts, solver, options,
                                          validation_tolerance)
        state_updates.append(diagnostics)
        if diagnostics["status"] != "eigenvalue":
            statuses.append(diagnostics["status"])
        return state

    for _ in range(restarts):
        measurements = []
        for dimension, inputs, outputs in zip(dimensions, game.scenario.input_sizes,
                                               game.scenario.output_sizes):
            party = []
            for _ in range(inputs):
                if measurement_class == "projective":
                    party.append(random_projective(rng, dimension, outputs))
                else:
                    gaussian = rng.normal(size=(outputs, dimension, dimension)) + 1j * rng.normal(
                        size=(outputs, dimension, dimension))
                    party.append(normalize_povm([g @ g.conj().T for g in gaussian]))
            measurements.append(tuple(party))
        measurements = tuple(measurements)
        state = update_state(measurements)
        strategy = QuantumStrategy(state, measurements, validation_tolerance)
        history = [strategy.score(game)]
        run_best, run_iteration, run_strategy = history[0], 0, strategy
        for iteration in range(1, max_iterations + 1):
            for party in range(parties):
                updated, local_statuses = optimize_party(coefficients, state, measurements,
                                                         dimensions, party, solver, options,
                                                         measurement_class=measurement_class)
                statuses.extend(local_statuses)
                measurements = measurements[:party] + (updated,) + measurements[party+1:]
            state = update_state(measurements)
            strategy = QuantumStrategy(state, measurements, validation_tolerance)
            score = strategy.score(game)
            history.append(score)
            if score > run_best:
                run_best, run_iteration, run_strategy = score, iteration, strategy
            if abs(history[-1] - history[-2]) <= convergence_tolerance:
                break
        restart_scores.append(run_best)
        if run_best > best_value:
            best_strategy, best_value = run_strategy, run_best
            best_history, best_iteration = tuple(history), run_iteration
    return SeesawResult(best_strategy, best_value, best_history, tuple(restart_scores),
                        best_iteration, int(seed), solver, tuple(statuses),
                        fixed_state is not None, cuts, tuple(state_updates), measurement_class=measurement_class)
