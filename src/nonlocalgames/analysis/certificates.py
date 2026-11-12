"""Exact post-verification of rationally reconstructed nonsignalling LP witnesses."""

from dataclasses import replace
from fractions import Fraction
from itertools import islice
from math import prod

from ..core.game import rational
from .nonsignalling import _nonsignalling_program, _solve_nonsignalling
from .result import AnalysisResult


def _verify_rational_lp(matrix, rhs, coefficients, primal, dual):
    """Verify primal and dual feasibility using rational arithmetic only.

    The LP is min -c.p subject to A.p=b and p>=0. Thus a dual witness y
    must obey A.T@y <= -c, giving the upper bound -b.y on the game score.
    """
    if len(primal) != len(coefficients) or len(dual) != len(rhs):
        raise ValueError("Certificate dimensions do not match the linear program")
    if any(value < 0 for value in primal):
        raise ValueError("Reconstructed primal certificate has a negative probability")
    dual_lhs = [Fraction(0)] * len(coefficients)
    for row, target in enumerate(rhs):
        value = Fraction(0)
        for offset in range(matrix.indptr[row], matrix.indptr[row + 1]):
            column = int(matrix.indices[offset])
            coefficient = int(matrix.data[offset])
            if coefficient != matrix.data[offset]:
                raise ValueError("Expected an integer constraint matrix")
            value += coefficient * primal[column]
            dual_lhs[column] += coefficient * dual[row]
        if value != target:
            raise ValueError("Reconstructed primal certificate violates an exact equality")
    if any(value > -coefficient for value, coefficient in zip(dual_lhs, coefficients)):
        raise ValueError("Reconstructed dual certificate violates an exact inequality")
    lower = sum((c*p for c, p in zip(coefficients, primal)), Fraction(0))
    upper = -sum((b*y for b, y in zip(rhs, dual)), Fraction(0))
    return lower, upper


def certify_nonsignalling(game, *, max_denominator: int = 1_000_000,
                         max_variables: int = 100_000) -> AnalysisResult:
    """Solve numerically, reconstruct rational witnesses, then verify them exactly.

    Reconstruction can fail even when the numerical optimum is accurate. A
    failed feasibility check raises ValueError; no certificate is then claimed.
    On success, value is the certified upper bound. If primal and dual scores
    agree, it is the exact optimum; otherwise diagnostics record the certified
    interval. No numerical solver status alone establishes certification.
    """
    if (isinstance(max_denominator, bool) or not isinstance(max_denominator, int)
            or max_denominator < 1):
        raise ValueError("max_denominator must be a positive integer")
    numerical, _, _, numeric_dual = _solve_nonsignalling(game, max_variables)
    primal = tuple(Fraction(float(v)).limit_denominator(max_denominator)
                   for v in numerical.witness.ravel())
    dual = tuple(Fraction(float(v)).limit_denominator(max_denominator) for v in numeric_dual)
    verified = verify_nonsignalling_certificate(game, primal, dual, max_variables=max_variables)
    return replace(verified, method="rational_lp_certificate", diagnostics={
        **verified.diagnostics, "max_denominator": max_denominator,
        "numerical_solver": numerical.diagnostics["solver"],
    })


def _certificate_dimensions(game, max_variables):
    """Preflight the v1 normalization/complement-marginal row counts."""
    if isinstance(max_variables, bool) or not isinstance(max_variables, int) or max_variables < 1:
        raise ValueError("max_variables must be a positive integer")
    inputs, outputs = game.scenario.input_sizes, game.scenario.output_sizes
    input_count, output_count = prod(inputs), prod(outputs)
    variables = input_count * output_count
    if variables > max_variables:
        raise ValueError(f"nonsignalling LP needs {variables} variables; limit is {max_variables}")
    equalities = input_count + sum((input_count // nx) * (output_count // na) * (nx - 1)
                                   for nx, na in zip(inputs, outputs))
    return variables, equalities


def _bounded_entries(values, expected, name):
    entries = tuple(islice(iter(values), expected + 1))
    if len(entries) != expected:
        raise ValueError(f"Certificate {name} dimensions do not match the linear program")
    return entries


def verify_nonsignalling_certificate(game, primal, dual, *, max_variables: int = 100_000):
    """Recheck an exported rational LP witness without solving an optimization.

    Accept exact integers, Fractions, or rational strings; floating-point data
    must first be reconstructed explicitly. Primal entries are flattened in
    input-major/output-minor order. Dual entries use the v1 constraint ordering
    documented by ``_nonsignalling_program``. No tolerance is used. On success,
    the value is the certified upper bound, with the lower bound in diagnostics;
    equal endpoints establish the exact optimum. Invalid witnesses raise.
    Size budgets and both vector lengths are checked before rational conversion
    or sparse-program allocation; at most the expected length plus one is read.
    """
    variables, equalities = _certificate_dimensions(game, max_variables)
    primal_entries = _bounded_entries(primal, variables, "primal")
    dual_entries = _bounded_entries(dual, equalities, "dual")
    primal, dual = tuple(map(rational, primal_entries)), tuple(map(rational, dual_entries))
    matrix, rhs = _nonsignalling_program(game, max_variables)
    if matrix.shape != (equalities, variables) or len(rhs) != equalities:
        raise RuntimeError("Nonsignalling certificate row schema differs from its preflight dimensions")
    lower, upper = _verify_rational_lp(matrix, rhs, game.coefficients, primal, dual)
    exact = lower == upper
    return AnalysisResult(
        upper, "nonsignalling", "exact_lp_verification",
        "exact" if exact else "certified_upper_bound",
        "optimal" if exact else "certified_interval",
        {"certified": True, "lower_bound": lower, "upper_bound": upper,
         "constraint_schema": "nonlocalgames.nonsignalling.v1"},
        {"primal": primal, "dual": dual},
    )
