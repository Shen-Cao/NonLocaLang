"""Numerical block updates used by fixed-dimensional strategy searches."""
from functools import reduce
from math import prod

import numpy as np


def normalize_povm(effects):
    """PSD projection followed by congruence normalization, without score repair."""
    projected = []
    for effect in effects:
        if not np.isfinite(effect).all():
            raise RuntimeError("The measurement SDP returned nonfinite effects")
        values, vectors = np.linalg.eigh((effect + effect.conj().T) / 2)
        projected.append((vectors * np.maximum(values, 0)) @ vectors.conj().T)
    values, vectors = np.linalg.eigh(sum(projected))
    if values.min() <= np.finfo(float).eps:
        raise RuntimeError("The measurement effects cannot be normalized stably")
    inverse_root = (vectors * (1 / np.sqrt(values))) @ vectors.conj().T
    return tuple(inverse_root @ effect @ inverse_root for effect in projected)


def bell_operator(game, measurements, dimensions):
    bell = np.zeros((prod(dimensions),) * 2, dtype=complex)
    for x in game.scenario.inputs:
        for a in game.scenario.outputs:
            operator = reduce(np.kron, (measurements[i][x[i]][a[i]]
                                       for i in range(len(dimensions))))
            bell += float(game.coefficient(x, a)) * operator
    return (bell + bell.conj().T) / 2


def optimize_party(coefficients, state, measurements, dimensions, party, solver, options):
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
        problem = cp.Problem(cp.Maximize(objective), [effect >> 0 for effect in effects]
                             + [sum(effects) == np.eye(dimensions[party])])
        problem.solve(solver=solver, **options)
        statuses.append(str(problem.status))
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or any(
                effect.value is None for effect in effects):
            raise RuntimeError(f"Measurement SDP did not return a usable solution: {problem.status}")
        updated.append(normalize_povm([effect.value for effect in effects]))
    return tuple(updated), statuses
