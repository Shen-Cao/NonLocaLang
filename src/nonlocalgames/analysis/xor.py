"""Tsirelson's vector SDP for bipartite binary-output XOR objectives.

Reference: Cleve et al., https://arxiv.org/abs/quant-ph/0404076.
"""

from ._sdp import solve_sdp
from .result import AnalysisResult


def xor_quantum_value(game, *, solver=None, max_matrix_size: int = 1000, **solver_options):
    """Optimize a payoff depending only on the output XOR, numerically.

    Weighted XOR rewards need not be Boolean. The vector SDP characterizes the
    unrestricted finite-dimensional tensor-product quantum value in this case.
    The returned Gram matrix is not itself a state-and-measurement realization.
    """
    import cvxpy as cp
    import numpy as np

    s = game.scenario
    if len(s.input_sizes) != 2 or s.output_sizes != (2, 2):
        raise ValueError("XOR SDP requires two parties with two outputs each")
    m, n = s.input_sizes
    if isinstance(max_matrix_size, bool) or not isinstance(max_matrix_size, int) or max_matrix_size < 1:
        raise ValueError("max_matrix_size must be a positive integer")
    if m + n > max_matrix_size:
        raise ValueError(f"XOR Gram matrix size {m + n} exceeds limit {max_matrix_size}")
    constant = 0
    weights = np.zeros((m, n))
    for x in s.inputs:
        even, odd = game.reward(x, (0, 0)), game.reward(x, (0, 1))
        if game.reward(x, (1, 1)) != even or game.reward(x, (1, 0)) != odd:
            raise ValueError("payoff must depend only on the XOR of the two answers")
        constant += game.probability(x) * (even + odd) / 2
        weights[x] = float(game.probability(x) * (even - odd) / 2)
    gram = cp.Variable((m + n, m + n), symmetric=True)
    constraints = [gram >> 0, cp.diag(gram) == 1]
    problem = cp.Problem(cp.Maximize(float(constant) + cp.sum(cp.multiply(weights, gram[:m, m:]))), constraints)
    value, witness, diagnostics = solve_sdp(problem, gram, constraints, solver, solver_options)
    diagnostics["mathematical_program"] = "exact characterization of unrestricted bipartite XOR quantum value"
    return AnalysisResult(value, "quantum_tensor_unbounded_dimension", "xor_vector_sdp",
                          "numerical_optimum", problem.status, diagnostics, witness)
