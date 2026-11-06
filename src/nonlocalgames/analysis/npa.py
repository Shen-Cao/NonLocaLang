"""Bipartite NPA relaxations with full projective measurement alphabets.

Reference: Navascues, Pironio and Acin, https://arxiv.org/abs/0803.4290.
The implementation builds Q1 or Q(1+AB), not an arbitrary-level hierarchy.
"""

from collections import Counter
from itertools import product

from ._sdp import solve_sdp
from .result import AnalysisResult


def reduce_word(word):
    """Reduce a projector word using cross-party commutation and orthogonality.

    A token is ``(party, question, answer)``. ``None`` denotes the zero word;
    the empty tuple denotes identity. No same-party commutation is assumed.
    """
    reduced = []
    for party in (0, 1):
        local = []
        for token in word:
            if token[0] != party:
                continue
            if local and token[1] == local[-1][1]:
                if token[2] != local[-1][2]:
                    return None
            else:
                local.append(token)
        reduced.extend(local)
    return tuple(reduced)


def _moment_key(word):
    reduced = reduce_word(word)
    if reduced is None:
        return None
    return min(reduced, reduce_word(tuple(reversed(reduced))))


def npa_bound(game, *, level="1+AB", solver=None, max_matrix_size: int = 300,
              **solver_options) -> AnalysisResult:
    """Numerically solve a quantum-commuting outer relaxation.

    Projectors include every output. Completeness, orthogonality, idempotence,
    and cross-party commutation are imposed at represented word lengths.
    Q1 also includes explicit nonnegative probabilities (sometimes called
    Q1 intersect NS). Real symmetric moments suffice for real Bell objectives:
    the real part of a complex feasible matrix is real PSD and obeys the same
    real linear equalities. This does not assume real physical measurements.

    The exact SDP optimum upper-bounds the commuting-operator value, but this
    floating-point result has no rigorous bound guarantee.
    """
    import cvxpy as cp
    import numpy as np
    from scipy.sparse import coo_matrix

    s = game.scenario
    if len(s.input_sizes) != 2:
        raise ValueError("NPA backend currently supports two parties")
    if level not in ("1", "1+AB"):
        raise ValueError("supported NPA levels are '1' and '1+AB'")
    if isinstance(max_matrix_size, bool) or not isinstance(max_matrix_size, int) or max_matrix_size < 1:
        raise ValueError("max_matrix_size must be a positive integer")
    na, nb = (s.input_sizes[i] * s.output_sizes[i] for i in (0, 1))
    size = 1 + na + nb + (na * nb if level == "1+AB" else 0)
    if size > max_matrix_size:
        raise ValueError(f"NPA moment matrix size {size} exceeds limit {max_matrix_size}")
    alice = [(0, x, a) for x in range(s.input_sizes[0]) for a in range(s.output_sizes[0])]
    bob = [(1, y, b) for y in range(s.input_sizes[1]) for b in range(s.output_sizes[1])]
    words = [()] + [(a,) for a in alice] + [(b,) for b in bob]
    if level == "1+AB":
        words += list(product(alice, bob))
    gamma = cp.Variable((size, size), symmetric=True)
    constraints = [gamma >> 0, gamma[0, 0] == 1]
    flat = cp.vec(gamma, order="C")
    moments = {}
    linear_rows = []
    for i, u in enumerate(words):
        for j in range(i, size):
            key = _moment_key(tuple(reversed(u)) + words[j])
            if key is None:
                linear_rows.append({i * size + j: 1})
            elif key in moments:
                linear_rows.append({i * size + j: 1, moments[key]: -1})
            else:
                moments[key] = i * size + j
    moment_indices = {key: i for i, key in enumerate(moments)}
    completeness = set()
    for word in moments:
        for position, token in enumerate(word):
            shorter = _moment_key(word[:position] + word[position + 1:])
            alternatives = [_moment_key(word[:position] + ((token[0], token[1], a),) + word[position + 1:])
                            for a in range(s.output_sizes[token[0]])]
            if shorter not in moments or any(k is not None and k not in moments for k in alternatives):
                continue
            coefficients = Counter(moment_indices[k] for k in alternatives if k is not None)
            coefficients[moment_indices[shorter]] -= 1
            relation = tuple(sorted((i, c) for i, c in coefficients.items() if c))
            if relation:
                completeness.add(relation)
    moment_vector = tuple(moments.values())
    linear_rows.extend({moment_vector[i]: c for i, c in relation}
                       for relation in sorted(completeness))
    if linear_rows:
        rows, columns, values = [], [], []
        for row, relation in enumerate(linear_rows):
            for column, value in relation.items():
                rows.append(row)
                columns.append(column)
                values.append(value)
        equalities = coo_matrix((values, (rows, columns)),
                               shape=(len(linear_rows), size * size)).tocsr()
        constraints.append(equalities @ flat == 0)
    probabilities = []
    coefficients = []
    for x in s.inputs:
        for a in s.outputs:
            probabilities.append(moments[_moment_key(((0, x[0], a[0]), (1, x[1], a[1])))])
            coefficients.append(float(game.coefficient(x, a)))
    probabilities = flat[probabilities]
    constraints.append(probabilities >= 0)
    problem = cp.Problem(cp.Maximize(np.asarray(coefficients) @ probabilities), constraints)
    value, witness, diagnostics = solve_sdp(problem, gamma, constraints, solver, solver_options)
    diagnostics.update({
        "level": level,
        "independent_real_moments": len(moments),
        "completeness_relations": len(completeness),
        "mathematical_program": "upper relaxation of quantum commuting value",
        "minimum_probability": float(np.min(probabilities.value)),
    })
    return AnalysisResult(value, "quantum_commuting", f"npa_{level}",
                          "numerical_relaxation", problem.status, diagnostics,
                          {"moment_matrix": witness, "words": tuple(words),
                           "behavior": np.asarray(probabilities.value).reshape(len(s.inputs), len(s.outputs))})
