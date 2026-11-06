"""Numerical block updates used by fixed-dimensional strategy searches."""
from functools import reduce
from itertools import product
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


def effective_operators(coefficients, state, measurements, dimensions, party):
    """Contract all other parties into local objective matrices, in explicit factor order.

    After moving this party first, rho has entries rho[i,r,k,s]. Contracting
    other-party effects O gives H[i,k]=sum_{r,s}rho[i,r,k,s]O[s,r], so the
    contribution of a local effect E is tr(E H). No real-state assumption is used.
    """
    count = len(dimensions)
    if coefficients.ndim != 2*count or not 0 <= party < count:
        raise ValueError("Coefficient tensor or party index does not match the tensor factors")
    input_sizes, output_sizes = coefficients.shape[:count], coefficients.shape[count:]
    others = tuple(i for i in range(count) if i != party)
    order = (party,) + others
    dimension, rest = dimensions[party], prod(dimensions[i] for i in others)
    rho = state.reshape(dimensions + dimensions).transpose(
        order + tuple(i+count for i in order)).reshape(dimension, rest, dimension, rest)
    effective = [[np.zeros((dimension, dimension), dtype=complex)
                  for _ in range(output_sizes[party])] for _ in range(input_sizes[party])]
    for other_x in product(*(range(input_sizes[i]) for i in others)):
        for other_a in product(*(range(output_sizes[i]) for i in others)):
            operator = reduce(np.kron, (measurements[i][x][a] for i, x, a in
                                        zip(others, other_x, other_a)), np.ones((1, 1)))
            marginal = np.einsum("irks,sr->ik", rho, operator, optimize=True)
            x, a = [0]*count, [0]*count
            for i, setting, output in zip(others, other_x, other_a):
                x[i], a[i] = setting, output
            for setting in range(input_sizes[party]):
                x[party] = setting
                for output in range(output_sizes[party]):
                    a[party] = output
                    effective[setting][output] += coefficients[tuple(x+a)]*marginal
    return tuple(tuple((h+h.conj().T)/2 for h in povm) for povm in effective)


def optimize_party(coefficients, state, measurements, dimensions, party, solver, options,
                   *, measurement_class="povm"):
    """Optimize one party while all other measurements and the state remain fixed."""
    from ._measurements import projective_update, validate_measurement_class
    measurement_class = validate_measurement_class(measurement_class, coefficients.shape[len(dimensions):])
    effective = effective_operators(coefficients, state, measurements, dimensions, party)
    if measurement_class == "projective":
        return tuple(projective_update(povm) for povm in effective), ["analytic_projective"]*len(effective)
    import cvxpy as cp

    updated, statuses = [], []
    for objective_matrices in effective:
        effects = [cp.Variable((1, 1), symmetric=True) if dimensions[party] == 1
                   else cp.Variable((dimensions[party],) * 2, hermitian=True) for _ in objective_matrices]
        objective_matrices = [h.real if dimensions[party] == 1 else h for h in objective_matrices]
        terms = [cp.sum(cp.multiply(effect, h.T)) for effect, h in zip(effects, objective_matrices)]
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
