"""Sparse linear programming over the full multipartite nonsignalling polytope."""

from itertools import product
from math import prod

from .result import AnalysisResult


def _nonsignalling_program(game, max_variables):
    """Construct integer equalities without invoking a numerical solver.

    Rows are normalization in input order, then party/complement-input/
    complement-output/nonzero-party-input in lexicographic order. Columns are
    input-major/output-minor. This ordering defines certificate schema v1.
    """
    from scipy.sparse import coo_matrix

    s = game.scenario
    nvars = prod(s.input_sizes) * prod(s.output_sizes)
    if isinstance(max_variables, bool) or not isinstance(max_variables, int) or max_variables < 1:
        raise ValueError("max_variables must be a positive integer")
    if nvars > max_variables:
        raise ValueError(f"nonsignalling LP needs {nvars} variables; limit is {max_variables}")
    inputs, outputs = tuple(s.inputs), tuple(s.outputs)
    xi, ai = {x: i for i, x in enumerate(inputs)}, {a: i for i, a in enumerate(outputs)}
    rows, cols, data, rhs = [], [], [], []

    def add_row(terms, value):
        row = len(rhs)
        for x, a, coefficient in terms:
            rows.append(row)
            cols.append(xi[x] * len(outputs) + ai[a])
            data.append(coefficient)
        rhs.append(value)

    for x in inputs:
        add_row(((x, a, 1) for a in outputs), 1)
    for party in range(len(s.input_sizes)):
        other_parties = tuple(i for i in range(len(s.input_sizes)) if i != party)
        for other_x in product(*(range(s.input_sizes[i]) for i in other_parties)):
            for other_a in product(*(range(s.output_sizes[i]) for i in other_parties)):
                x = [0] * len(s.input_sizes)
                a = [0] * len(s.input_sizes)
                for i, value in zip(other_parties, other_x):
                    x[i] = value
                for i, value in zip(other_parties, other_a):
                    a[i] = value
                for question in range(1, s.input_sizes[party]):
                    terms = []
                    for answer in range(s.output_sizes[party]):
                        a[party] = answer
                        x[party] = question
                        terms.append((tuple(x), tuple(a), 1))
                        x[party] = 0
                        terms.append((tuple(x), tuple(a), -1))
                    add_row(terms, 0)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), nvars)).tocsr()
    return matrix, tuple(rhs)


def _solve_nonsignalling(game, max_variables):
    import numpy as np
    from scipy.optimize import linprog

    matrix, rhs = _nonsignalling_program(game, max_variables)
    b = np.asarray(rhs, dtype=float)
    c = np.asarray([float(value) for value in game.coefficients])
    solved = linprog(-c, A_eq=matrix, b_eq=b, bounds=(0, None), method="highs")
    if not solved.success or solved.x is None:
        raise RuntimeError(f"nonsignalling LP failed: {solved.message}")
    dual = np.asarray(solved.eqlin.marginals)
    dual_slack = -c - matrix.T @ dual
    diagnostics = {
        "solver": "scipy.highs", "variables": matrix.shape[1], "equalities": len(rhs),
        "max_equality_residual": float(np.max(np.abs(matrix @ solved.x - b))),
        "minimum_probability": float(np.min(solved.x)),
        "minimum_dual_slack": float(np.min(dual_slack)),
        "primal_dual_gap": float(solved.fun - b @ dual),
        "certified": False,
    }
    result = AnalysisResult(float(-solved.fun), "nonsignalling", "sparse_lp",
                            "numerical_optimum", "optimal", diagnostics,
                            solved.x.reshape(len(game.inputs), len(game.outputs)))
    return result, matrix, rhs, dual


def nonsignalling_value(game, *, max_variables: int = 100_000) -> AnalysisResult:
    """Return a numerical optimum and behavior, with primal/dual residuals.

    All inputs, including zero-probability questions, remain in the constraints.
    Requiring the marginal of every party's complement to be independent of that
    party's question is equivalent to all subset nonsignalling constraints.
    """
    return _solve_nonsignalling(game, max_variables)[0]
