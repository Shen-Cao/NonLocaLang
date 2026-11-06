"""Numerical XOR dual discovery followed by bounded exact rational repair."""
from dataclasses import replace
from fractions import Fraction
from numbers import Integral

from .linalg import positive_integer, verify_psd
from .xor import dual_slack, verify_xor_dual
from .xor_model import cost_matrix, xor_coefficients


def _is_psd(matrix):
    try:
        verify_psd(matrix, max_size=max(1, len(matrix)))
    except ValueError:
        return False
    return True


def repair_diagonal(matrix, diagonal, steps):
    """Find an exactly feasible uniform shift with a guaranteed Gershgorin endpoint."""
    if _is_psd(dual_slack(matrix, diagonal)):
        return tuple(diagonal), Fraction(0), Fraction(0), 0
    endpoint = max(Fraction(0), max(sum(abs(v) for j, v in enumerate(row) if i != j)
                                  - diagonal[i] + row[i] for i, row in enumerate(matrix)))
    lower, upper = Fraction(0), endpoint
    candidate = tuple(value + upper for value in diagonal)
    if not _is_psd(dual_slack(matrix, candidate)):
        raise RuntimeError("Exact diagonal-dominance repair failed its PSD invariant")
    for _ in range(steps):
        midpoint = (lower + upper) / 2
        candidate = tuple(value + midpoint for value in diagonal)
        if _is_psd(dual_slack(matrix, candidate)):
            upper = midpoint
        else:
            lower = midpoint
    return tuple(value + upper for value in diagonal), upper, endpoint, steps


def certify_xor_upper_bound(game, *, solver=None, max_denominator=1_000_000,
                            repair_steps=30, max_matrix_size=64, **solver_options):
    """Discover a numerical dual, reconstruct it, and independently verify its bound.

    Rational reconstruction alone need not preserve PSD. A uniform rational
    diagonal shift repairs the slack; a bounded bisection uses exact PSD tests
    to reduce the shift from a guaranteed diagonal-dominance endpoint. The
    resulting bound, not the raw numerical objective, is the certified value.
    Solver failure propagates; it does not silently select a different solver
    or replace the requested optimization with a trivial bound.
    """
    from math import isfinite
    from ..analysis.xor import _solve_xor
    denominator = positive_integer(max_denominator, "max_denominator")
    if isinstance(repair_steps, bool) or not isinstance(repair_steps, Integral) or repair_steps < 0:
        raise ValueError("repair_steps must be a nonnegative integer")
    constant, weights = xor_coefficients(game, max_matrix_size=max_matrix_size)
    numerical, dual = _solve_xor(game, solver=solver, max_matrix_size=max_matrix_size, **solver_options)
    if dual is None or len(dual) != sum(game.scenario.input_sizes) or any(not isfinite(float(v)) for v in dual):
        raise RuntimeError("XOR solver returned a missing or nonfinite dual diagonal")
    numerical_dual_objective = float(constant) + sum(float(value) for value in dual)
    if not isfinite(numerical_dual_objective):
        raise RuntimeError("XOR numerical dual objective exceeds the finite floating range")
    diagonal = tuple(Fraction(float(value)).limit_denominator(denominator) for value in dual)
    matrix = cost_matrix(weights)
    repaired, shift, endpoint, steps = repair_diagonal(matrix, diagonal, int(repair_steps))
    checked = verify_xor_dual(game, repaired, max_matrix_size=max_matrix_size)
    return replace(checked, method="rational_xor_dual_certificate", diagnostics={
        **checked.diagnostics, "numerical_solver": numerical.diagnostics["solver"],
        "numerical_status": numerical.status, "numerical_primal_objective": numerical.value,
        "numerical_dual_objective": numerical_dual_objective,
        "reconstructed_dual_objective": constant + sum(diagonal, Fraction(0)),
        "diagonal_shift": shift, "repair_bound_increase": len(diagonal)*shift,
        "gershgorin_shift_endpoint": endpoint, "repair_steps": steps,
        "max_denominator": denominator,
    })
