"""Construct a new exact finite quantum strategy from numerical matrix data."""
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, prod
from numbers import Integral
from typing import TYPE_CHECKING

from ..strategy import QuantumStrategy
from .algebra import add, identity, scale, subtract
from .budgets import ExactBudgetExceeded, coerce_budget
from .matrix import ExactPSDViolation, RationalMatrix, verify_psd_complex

if TYPE_CHECKING:
    from .strategy import ExactQuantumStrategy


@dataclass(frozen=True)
class RationalizationResult:
    """A new exact realization and reconstruction diagnostics, not an optimality claim."""
    strategy: "ExactQuantumStrategy"
    diagnostics: dict


def _preflight(strategy, budget):
    dimensions = tuple(strategy.dimensions)
    if not dimensions or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1
                             for d in dimensions):
        raise ValueError("Numerical strategy dimensions must be positive integers")
    if len(dimensions) > budget.max_parties:
        raise ExactBudgetExceeded("Rationalization exceeds max_parties")
    size = prod(dimensions)
    if size > budget.max_joint_dimension:
        raise ExactBudgetExceeded("Rationalization exceeds max_joint_dimension")
    if 2*size > budget.max_psd_size:
        raise ExactBudgetExceeded("Rationalization realification exceeds max_psd_size")
    if strategy.state.shape != (size, size) or len(strategy.measurements) != len(dimensions):
        raise ValueError("Numerical strategy state or parties do not match the tensor dimensions")
    entries = size*size
    for party, dimension in zip(strategy.measurements, dimensions):
        if not party or any(not povm for povm in party):
            raise ValueError("Numerical measurements must have settings and outcomes")
        if any(len(povm) != len(party[0]) for povm in party):
            raise ValueError("Rationalization requires a fixed output alphabet across each party's inputs")
        for povm in party:
            entries += len(povm)*dimension*dimension
            if entries > budget.max_matrix_entries:
                raise ExactBudgetExceeded("Rationalization exceeds max_matrix_entries")
            if any(effect.shape != (dimension, dimension) for effect in povm):
                raise ValueError("Numerical POVM effect shape does not match its party dimension")
    return tuple(map(int, dimensions)), entries


def _round_hermitian(array, denominator, budget):
    dimension = array.shape[0]
    real = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    imag = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    rounding_error = Fraction(0)
    for i in range(dimension):
        for j in range(dimension):
            for part, value in ((real, array[i, j].real), (imag, array[i, j].imag)):
                value = float(value)
                if not isfinite(value):
                    raise ValueError("Numerical strategy matrices must be finite")
                exact = Fraction(value)
                rounded = exact.limit_denominator(denominator)
                part[i][j] = rounded
                rounding_error = max(rounding_error, abs(exact-rounded))
    hermitian_real = tuple(tuple((real[i][j]+real[j][i])/2 for j in range(dimension))
                           for i in range(dimension))
    hermitian_imag = tuple(tuple((imag[i][j]-imag[j][i])/2 for j in range(dimension))
                           for i in range(dimension))
    correction = max((abs(before[i][j]-after[i][j])
                      for before, after in ((real, hermitian_real), (imag, hermitian_imag))
                      for i in range(dimension) for j in range(dimension)), default=Fraction(0))
    return RationalMatrix(hermitian_real, hermitian_imag, budget=budget), rounding_error, correction


def _component_max(matrix):
    return max((abs(value) for part in (matrix.real, matrix.imag) for row in part for value in row),
               default=Fraction(0))


def _mix_to_psd(matrices, target, steps, budget):
    """Use one common rational mixing weight and keep an exactly feasible endpoint."""
    checks = 0

    def positive(candidates):
        nonlocal checks
        try:
            for matrix in candidates:
                checks += 1
                verify_psd_complex(matrix, budget=budget)
        except ExactPSDViolation:
            return False
        return True

    if positive(matrices):
        return tuple(matrices), Fraction(0), 0, checks
    feasible = (target,)*len(matrices)
    if not positive(feasible):
        raise RuntimeError("Identity mixing endpoint failed exact positivity verification")
    lower, upper = Fraction(0), Fraction(1)
    for _ in range(steps):
        midpoint = (lower+upper)/2
        candidates = tuple(add(scale(matrix, 1-midpoint, budget=budget),
                               scale(target, midpoint, budget=budget), budget=budget)
                           for matrix in matrices)
        if positive(candidates):
            upper, feasible = midpoint, candidates
        else:
            lower = midpoint
    return feasible, upper, steps, checks


def rationalize_strategy(strategy, *, max_denominator=1_000_000, repair_steps=30, budget=None):
    """Round and repair a numerical QuantumStrategy into a new exact realization.

    Trace is corrected by a scalar identity shift; each POVM receives a common
    completeness correction. Exact PSD tests then select identity-mixture
    weights, common to every outcome of a POVM. The endpoint t=1 is exactly
    feasible even when no smaller candidate passes. No floating eigensolver is
    used to certify positivity, and budget errors never trigger a repair fallback.

    This preserves dimensions and the input/output interface, but does not promise
    closeness, score preservation, global optimality, a fixed state, PPT, projectivity,
    product structure, or network source independence. Network witnesses must not
    be implicitly flattened through this API. Source matrices remain unchanged.
    """
    from .strategy import ExactQuantumStrategy
    if not isinstance(strategy, QuantumStrategy):
        raise TypeError("Rationalization requires a plain numerical QuantumStrategy; network flattening is not implicit")
    if isinstance(max_denominator, bool) or not isinstance(max_denominator, Integral) or max_denominator < 1:
        raise ValueError("max_denominator must be a positive integer")
    if isinstance(repair_steps, bool) or not isinstance(repair_steps, Integral) or repair_steps < 0:
        raise ValueError("repair_steps must be a nonnegative integer")
    budget = coerce_budget(budget)
    dimensions, entries = _preflight(strategy, budget)
    size = prod(dimensions)
    state, error, correction = _round_hermitian(strategy.state, int(max_denominator), budget)
    trace, imaginary_trace = state.trace()
    if imaginary_trace != 0:
        raise RuntimeError("Hermitian rounding produced a nonreal trace")
    shift = (1-trace)/size
    state = add(state, scale(identity(size, budget=budget), shift, budget=budget), budget=budget)
    repaired, mixing, iterations, checks = _mix_to_psd(
        (state,), scale(identity(size, budget=budget), Fraction(1, size), budget=budget),
        int(repair_steps), budget)
    state_diagnostics = {"rounding_error_max_component": error, "hermitian_correction_max_component": correction,
                         "trace_before_correction": trace, "trace_identity_shift": shift,
                         "mixing_weight": mixing, "bisection_steps": iterations, "psd_checks": checks}
    measurements, diagnostics = [], []
    for party, dimension in zip(strategy.measurements, dimensions):
        exact_party, party_diagnostics = [], []
        unit = identity(dimension, budget=budget)
        for povm in party:
            rounded = tuple(_round_hermitian(effect, int(max_denominator), budget) for effect in povm)
            total = scale(unit, 0, budget=budget)
            for matrix, _, _ in rounded:
                total = add(total, matrix, budget=budget)
            correction = scale(subtract(unit, total, budget=budget), Fraction(1, len(povm)), budget=budget)
            corrected = tuple(add(matrix, correction, budget=budget) for matrix, _, _ in rounded)
            repaired_povm, weight, iterations, checks = _mix_to_psd(
                corrected, scale(unit, Fraction(1, len(povm)), budget=budget), int(repair_steps), budget)
            exact_party.append(repaired_povm)
            party_diagnostics.append({"rounding_error_max_component": max(value[1] for value in rounded),
                                      "hermitian_correction_max_component": max(value[2] for value in rounded),
                                      "completeness_correction_max_component": _component_max(correction),
                                      "mixing_weight": weight, "bisection_steps": iterations, "psd_checks": checks})
        measurements.append(tuple(exact_party))
        diagnostics.append(tuple(party_diagnostics))
    exact = ExactQuantumStrategy(repaired[0], tuple(measurements), dimensions=dimensions, budget=budget)
    return RationalizationResult(exact, {"max_denominator": int(max_denominator), "repair_steps": int(repair_steps),
        "local_dimensions": dimensions, "matrix_entries": entries, "state": state_diagnostics,
        "measurements": tuple(diagnostics), "scope": "new exact standard tensor-product realization",
        "restrictions_not_preserved": ("fixed_state", "ppt", "projective", "product", "network"),
        "closeness_or_optimality_claimed": False})
