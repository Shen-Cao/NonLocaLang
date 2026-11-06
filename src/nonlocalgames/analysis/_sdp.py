"""Shared numerical SDP checks; residuals are diagnostics, not certificates."""


def solve_sdp(problem, matrix, constraints, solver, solver_options):
    import cvxpy as cp
    import numpy as np

    problem.solve(solver=solver, **solver_options)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP did not return an optimum: {problem.status}")
    if matrix.value is None or problem.value is None or not np.isfinite(problem.value):
        raise RuntimeError("SDP returned missing or nonfinite numerical values")
    value = np.asarray(matrix.value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise RuntimeError("SDP returned a nonfinite matrix")
    diagnostics = {
        "solver": problem.solver_stats.solver_name,
        "iterations": problem.solver_stats.num_iters,
        "solve_time": problem.solver_stats.solve_time,
        "matrix_size": int(value.shape[0]),
        "minimum_eigenvalue": float(np.linalg.eigvalsh((value + value.T) / 2).min()),
        "max_constraint_violation": float(max((np.max(c.violation()) for c in constraints), default=0)),
        "certified": False,
        "inaccurate": problem.status == cp.OPTIMAL_INACCURATE,
    }
    return float(problem.value), value, diagnostics
