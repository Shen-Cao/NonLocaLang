"""Solver-independent exact dual upper bounds for bipartite XOR games.

For c0,W in the score convention and C=[[0,W/2],[W.T/2,0]], a rational z
with diag(z)-C PSD gives the upper bound c0+sum(z). Weak duality suffices;
no solver status, floating eigenvalue, or numerical primal witness is trusted.
"""
from fractions import Fraction
from itertools import islice

from ..core.game import rational
from .linalg import verify_psd
from .xor_model import cost_matrix, game_fingerprint, xor_coefficients

XOR_DUAL_SCHEMA = "nonlocalgames.xor_dual.v1"


def dual_slack(matrix, diagonal):
    return tuple(tuple((diagonal[i] if i == j else Fraction(0)) - value
                       for j, value in enumerate(row)) for i, row in enumerate(matrix))


def verify_xor_dual(game, diagonal, *, max_matrix_size=128):
    """Verify a rational dual diagonal and return its certified quantum upper bound.

    Entries may be Fractions, integers, or rational strings; implicit floating
    reconstruction is forbidden. The value bounds unrestricted finite-dimensional
    and commuting bipartite XOR strategies. No quantum lower bound or optimizer
    is inferred from a numerical Gram matrix. Signed parity rewards are allowed.
    """
    from ..analysis.result import AnalysisResult
    constant, weights = xor_coefficients(game, max_matrix_size=max_matrix_size)
    size = sum(game.scenario.input_sizes)
    entries = tuple(islice(iter(diagonal), size + 1))
    if len(entries) != size:
        raise ValueError("XOR dual diagonal length does not match the Gram matrix")
    diagonal = tuple(map(rational, entries))
    matrix = cost_matrix(weights)
    verified = verify_psd(dual_slack(matrix, diagonal), max_size=max_matrix_size)
    upper = constant + sum(diagonal, Fraction(0))
    payload = {"schema": XOR_DUAL_SCHEMA, "game_fingerprint": game_fingerprint(game),
               "diagonal": list(map(str, diagonal)), "upper_bound": str(upper)}
    return AnalysisResult(upper, "quantum_tensor_unbounded_dimension", "exact_xor_dual_verification",
                          "certified_upper_bound", "verified",
                          {"certified": True, "upper_bound": upper, "matrix_size": len(matrix),
                           "psd_rank": verified.rank, "psd_pivots": verified.pivots,
                           "scope": "upper bound on unrestricted bipartite XOR quantum value",
                           "certificate_schema": XOR_DUAL_SCHEMA}, payload)


def verify_xor_certificate(game, certificate, *, max_matrix_size=128):
    """Recheck a saved certificate, its verifier fingerprint, and its claimed bound."""
    fields = {"schema", "game_fingerprint", "diagonal", "upper_bound"}
    if not isinstance(certificate, dict) or set(certificate) != fields:
        raise ValueError("Invalid XOR certificate fields")
    if certificate["schema"] != XOR_DUAL_SCHEMA:
        raise ValueError("Unsupported XOR certificate schema")
    # Check applicability and dimensions before scanning potentially large payloads.
    xor_coefficients(game, max_matrix_size=max_matrix_size)
    if not isinstance(certificate["diagonal"], list) or len(certificate["diagonal"]) != sum(game.scenario.input_sizes):
        raise ValueError("Serialized XOR diagonal length does not match the Gram matrix")
    if any(not isinstance(value, str) for value in certificate["diagonal"]):
        raise ValueError("Serialized XOR diagonal entries must be rational strings")
    if not isinstance(certificate["upper_bound"], str):
        raise ValueError("Serialized XOR upper bound must be a rational string")
    if certificate["game_fingerprint"] != game_fingerprint(game):
        raise ValueError("XOR certificate is bound to a different verifier")
    result = verify_xor_dual(game, certificate["diagonal"], max_matrix_size=max_matrix_size)
    if result.value != rational(certificate["upper_bound"]):
        raise ValueError("Claimed XOR upper bound differs from the verified dual objective")
    return result
