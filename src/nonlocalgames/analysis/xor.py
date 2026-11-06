"""Tsirelson's vector SDP for bipartite binary-output XOR objectives.

Reference: Cleve et al., https://arxiv.org/abs/quant-ph/0404076.
"""
from ._sdp import solve_sdp
from .result import AnalysisResult
from ..certificates.xor_model import xor_coefficients


def _solve_xor(game, *, solver=None, max_matrix_size=1000, **solver_options):
    """Return the numerical result and diagonal-equality dual for certificate discovery."""
    import cvxpy as cp
    import numpy as np
    constant, exact_weights = xor_coefficients(game, max_matrix_size=max_matrix_size)
    try:
        offset = float(constant)
        weights = np.array([[float(value) for value in row] for row in exact_weights])
    except (ValueError, OverflowError) as error:
        raise ValueError("XOR coefficients cannot be represented in float64; rescale the game") from error
    if not np.isfinite(offset) or not np.isfinite(weights).all():
        raise ValueError("XOR coefficients exceed the finite float64 range")
    if (constant != 0 and offset == 0) or any(q != 0 and value == 0
            for exact_row, numeric_row in zip(exact_weights, weights)
            for q, value in zip(exact_row, numeric_row)):
        raise ValueError("XOR coefficients underflow float64; rescale the game or supply an exact dual")
    m, n = game.scenario.input_sizes
    gram = cp.Variable((m + n, m + n), symmetric=True)
    constraints = [gram >> 0, cp.diag(gram) == 1]
    problem = cp.Problem(cp.Maximize(offset + cp.sum(cp.multiply(weights, gram[:m, m:]))), constraints)
    value, witness, diagnostics = solve_sdp(problem, gram, constraints, solver, solver_options)
    diagnostics["mathematical_program"] = "exact characterization of unrestricted bipartite XOR quantum value"
    result = AnalysisResult(value, "quantum_tensor_unbounded_dimension", "xor_vector_sdp",
                            "numerical_optimum", problem.status, diagnostics, witness)
    raw_dual = constraints[1].dual_value
    dual = None if raw_dual is None else np.asarray(raw_dual, dtype=float).reshape(-1)
    return result, dual


def xor_quantum_value(game, *, solver=None, max_matrix_size: int = 1000, **solver_options):
    """Optimize a payoff depending only on the output XOR, numerically.

    Weighted XOR rewards need not be Boolean. Unused question tuples may have
    arbitrary payoffs; all local question axes remain in the Gram matrix.
    The vector SDP characterizes the
    unrestricted finite-dimensional tensor-product quantum value in this case.
    The returned Gram matrix is not itself a state-and-measurement realization.
    """
    return _solve_xor(game, solver=solver, max_matrix_size=max_matrix_size, **solver_options)[0]
