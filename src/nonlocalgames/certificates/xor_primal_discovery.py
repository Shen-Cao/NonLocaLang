"""Numerical Gram discovery with rational rounding and exact PSD repair."""
from dataclasses import replace
from fractions import Fraction
from numbers import Integral

from ..core.scalar_wire import parse_rational
from .linalg import positive_integer
from .xor import _preflight_game
from .xor_discovery import _is_psd
from .xor_primal import verify_xor_gram


def _round_gram(candidate, size, denominator, max_scalar_bits):
    import numpy as np
    try:
        matrix = np.asarray(candidate)
    except (TypeError, ValueError) as error:
        raise RuntimeError('XOR solver returned a malformed primal Gram candidate') from error
    if matrix.shape != (size, size) or matrix.dtype.kind not in 'iuf':
        raise RuntimeError('XOR solver must return a square real numerical primal Gram')
    with np.errstate(over='ignore', invalid='ignore'):
        matrix = matrix.astype(float, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError('XOR solver returned a nonfinite primal Gram candidate')
    rounded = [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(i):
            mean = (Fraction(float(matrix[i, j])) + Fraction(float(matrix[j, i]))) / 2
            entry = Fraction(round(denominator * mean), denominator)
            rounded[i][j] = rounded[j][i] = parse_rational(entry, max_bits=max_scalar_bits)
    return tuple(map(tuple, rounded))


def _mix_with_identity(gram, fraction):
    return tuple(tuple(Fraction(1) if i == j else (1-fraction)*value
                       for j, value in enumerate(row)) for i, row in enumerate(gram))


def _repair(gram, steps):
    if _is_psd(gram):
        return gram, Fraction(0), 0
    radius = max(sum(abs(v) for j, v in enumerate(row) if i != j) for i, row in enumerate(gram))
    # Unit diagonal and maximum absolute off-diagonal row sum ≤1 imply PSD.
    if radius <= 1:
        raise RuntimeError('A diagonally dominant rounded Gram failed its exact PSD check')
    lower, upper = Fraction(0), 1 - 1/radius
    feasible = _mix_with_identity(gram, upper)
    if not _is_psd(feasible):
        raise RuntimeError('Gram repair endpoint failed its exact PSD check')
    for _ in range(steps):
        midpoint = (lower + upper) / 2
        candidate = _mix_with_identity(gram, midpoint)
        if _is_psd(candidate):
            upper, feasible = midpoint, candidate
        else:
            lower = midpoint
    return feasible, upper, steps


def certify_xor_lower_bound(game, *, solver=None, max_denominator=1_000_000,
                            repair_steps=24, max_matrix_size=64, max_scalar_bits=4096,
                            max_scalar_chars=8192, **solver_options):
    """Round a numerical XOR Gram to a common-denominator grid and check its score.

    Fix the diagonal to one and round the average of each symmetric off-diagonal
    pair to the nearest grid point, rounding ties to even. A mixture with identity
    restores positive semidefiniteness when necessary. The final Gram has an
    exact independent PSD check and an exactly recomputed attained score.
    """
    from ..analysis.xor import _solve_xor
    denominator = positive_integer(max_denominator, 'max_denominator')
    if isinstance(repair_steps, bool) or not isinstance(repair_steps, Integral) or repair_steps < 0:
        raise ValueError('repair_steps must be a nonnegative integer')
    size = _preflight_game(game, max_matrix_size, max_scalar_bits, max_scalar_chars)
    numerical, _ = _solve_xor(game, solver=solver, max_matrix_size=max_matrix_size, **solver_options)
    gram = _round_gram(numerical.witness, size, denominator, max_scalar_bits)
    repaired, fraction, steps = _repair(gram, int(repair_steps))
    result = verify_xor_gram(game, repaired, max_matrix_size=max_matrix_size,
                             max_scalar_bits=max_scalar_bits, max_scalar_chars=max_scalar_chars)
    return replace(result, method='rational_xor_primal_certificate', diagnostics={
        **result.diagnostics, 'numerical_solver': numerical.diagnostics['solver'],
        'numerical_status': numerical.status, 'numerical_primal_objective': numerical.value,
        'rounding_denominator': denominator, 'identity_mixture': fraction, 'repair_steps': steps,
    })
