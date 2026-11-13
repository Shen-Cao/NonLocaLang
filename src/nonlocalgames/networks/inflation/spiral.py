"""Portable classical triangle spiral-inflation discovery with exact post-checks."""
from dataclasses import replace
from fractions import Fraction
from math import isfinite, prod
from numbers import Integral, Real

import numpy as np

from ...core.scalar_wire import ScalarBudgetExceeded
from ..behavior import positive_budget
from ._program import compile_spiral
from .certificates import (_dual_quantities, make_certificate, verify_spiral_certificate)
from .result import SpiralResult


def _candidate(vector, size, denominator):
    if vector is None:
        return None, None
    try:
        raw = np.asarray(vector)
        if raw.ndim != 1 or len(raw) != size or np.iscomplexobj(raw):
            return None, None
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None, None
    if not np.isfinite(values).all():
        return None, None
    rational = tuple(Fraction(float(value)).limit_denominator(denominator) for value in values)
    return rational, values


def _positive_real(value, name, minimum=0, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(numeric) or (numeric < 0 if allow_zero else numeric <= 0) or numeric < minimum:
        raise ValueError(f"{name} must be positive, finite, and at least {minimum}")
    return numeric


def _solver_options(options):
    options = dict(options or {})
    allowed = {"presolve", "disp", "time_limit", "maxiter", "dual_feasibility_tolerance",
               "primal_feasibility_tolerance", "ipm_optimality_tolerance"}
    if any(not isinstance(key, str) for key in options) or set(options) - allowed:
        raise ValueError("Unsupported HiGHS solver option")
    for name, value in options.items():
        if name in ("presolve", "disp"):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a Boolean")
            options[name] = bool(value)
        elif name == "maxiter":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= 2**31-1:
                raise ValueError("maxiter must be an integer between zero and 2**31-1")
            options[name] = int(value)
        else:
            minimum = 1e-12 if name == "ipm_optimality_tolerance" else (
                1e-10 if name.endswith("feasibility_tolerance") else 0)
            options[name] = _positive_real(value, name, minimum, allow_zero=name == "time_limit")
    return options


def triangle_spiral(network, behavior, *, max_denominator=1_000_000, solver_options=None,
                    validation_tolerance=1e-7,
                    max_variables=100_000, max_equalities=100_000, max_nonzeros=500_000,
                    max_scalar_bits=4096, max_scalar_chars=8192):
    """Test one classical fanout inflation, with exactly verified rational evidence.

    A positive primal certificate proves only feasibility of this relaxation.
    A Farkas certificate proves classical incompatibility; it never excludes
    quantum sources. Inaccurate numerical outcomes without verified certificates
    remain not_ruled_out or unknown. No alternate solver is selected silently.
    Only no-input triangles are accepted. The sparse size guards apply before
    any inflated assignments are enumerated.
    """
    from scipy.optimize import linprog
    denominator = positive_budget(max_denominator, "max_denominator")
    options = _solver_options(solver_options)
    tolerance = _positive_real(validation_tolerance, "validation_tolerance")
    budgets = dict(max_variables=max_variables, max_equalities=max_equalities, max_nonzeros=max_nonzeros,
                   max_scalar_bits=max_scalar_bits, max_scalar_chars=max_scalar_chars)
    program = compile_spiral(network, behavior, **budgets)
    matrix = program.matrix
    rhs = np.asarray(program.rhs, dtype=float)
    diagnostics = {**program.sizes, "solver": "scipy.highs", "solver_options": options,
                   "max_denominator": denominator, "validation_tolerance": tolerance, "certified": False,
                   "rhs_underflow_entries": sum(q != 0 and f == 0 for q, f in zip(program.rhs, rhs)),
                   "scope": "fixed classical fanout spiral relaxation; no quantum inference"}
    primal = linprog(np.zeros(matrix.shape[1]), A_eq=matrix, b_eq=rhs,
                     bounds=(0, None), method="highs", options=options)
    diagnostics.update(primal_status=getattr(primal, "status", None),
                       primal_message=str(getattr(primal, "message", "missing solver message")))
    primal_candidate, primal_values = _candidate(getattr(primal, "x", None), matrix.shape[1], denominator)
    if primal_candidate is not None:
        try:
            checked = verify_spiral_certificate(network, behavior,
                make_certificate(program, "primal", primal_candidate), **budgets)
            return replace(checked, diagnostics={**diagnostics, **checked.diagnostics})
        except ScalarBudgetExceeded:
            raise
        except ValueError as error:
            diagnostics["primal_reconstruction_error"] = str(error)
        diagnostics["numerical_primal_minimum"] = float(np.min(primal_values))
        with np.errstate(over="ignore", invalid="ignore"):
            residual = float(np.max(np.abs(matrix @ primal_values - rhs)))
        diagnostics["numerical_primal_max_residual"] = residual if isfinite(residual) else None
        if not isfinite(residual):
            diagnostics["numerical_primal_error"] = "nonfinite residual from supplied solver vector"
    # Bounded dual discovery is always feasible (y=0) and bounded. Search bounds
    # are not certificate assumptions: any genuine Farkas vector can be scaled.
    dual = linprog(rhs, A_ub=-matrix.T, b_ub=np.zeros(matrix.shape[1]),
                   bounds=(-1, 1), method="highs", options=options)
    diagnostics.update(dual_status=getattr(dual, "status", None),
                       dual_message=str(getattr(dual, "message", "missing solver message")))
    dual_candidate, _ = _candidate(getattr(dual, "x", None), matrix.shape[0], denominator)
    if dual_candidate is not None:
        slacks, _ = _dual_quantities(program, dual_candidate)
        shift = max(Fraction(0), -min(slacks))
        first_block = prod(program.output_sizes)
        repaired = tuple(value + shift if i < first_block else value
                         for i, value in enumerate(dual_candidate))
        diagnostics["dual_repair_shift"] = shift
        try:
            checked = verify_spiral_certificate(network, behavior,
                make_certificate(program, "dual", repaired), **budgets)
            return replace(checked, diagnostics={**diagnostics, **checked.diagnostics})
        except ScalarBudgetExceeded:
            raise
        except ValueError as error:
            diagnostics["dual_reconstruction_error"] = str(error)
    if (getattr(primal, "status", None) == 0 and primal_candidate is not None
            and diagnostics["numerical_primal_minimum"] >= -tolerance
            and diagnostics["numerical_primal_max_residual"] is not None
            and diagnostics["numerical_primal_max_residual"] <= tolerance):
        return SpiralResult("not_ruled_out", "numerical_relaxation_feasibility", diagnostics)
    return SpiralResult("unknown", "uncertified_solver_output", diagnostics)
