"""Solver-independent exact dual upper bounds for bipartite XOR games.

For c0,W in the score convention and C=[[0,W/2],[W.T/2,0]], a rational z
with diag(z)-C PSD gives the upper bound c0+sum(z). Weak duality suffices;
no solver status, floating eigenvalue, or numerical primal witness is trusted.
"""
from fractions import Fraction
from itertools import islice

from ..core.game import Game
from ..core.scalar_wire import parse_rational, scalar_string
from ..core.wire_validation import validate_game_numbers
from .linalg import _verify_psd_prepared, positive_integer
from .xor_model import cost_matrix, game_fingerprint, xor_coefficients

XOR_DUAL_SCHEMA = "nonlocalgames.xor_dual.v1"


def dual_slack(matrix, diagonal):
    return tuple(tuple((diagonal[i] if i == j else Fraction(0)) - value
                       for j, value in enumerate(row)) for i, row in enumerate(matrix))


def _preflight_game(game, max_matrix_size, max_scalar_bits, max_scalar_chars):
    if not isinstance(game, Game):
        raise TypeError("Expected a finite Game")
    scenario = game.scenario
    if scenario.parties != 2 or scenario.output_sizes != (2, 2):
        raise ValueError("XOR SDP requires two parties with two outputs each")
    size = sum(scenario.input_sizes)
    limit = positive_integer(max_matrix_size, "max_matrix_size")
    if size > limit:
        raise ValueError(f"XOR Gram matrix size {size} exceeds limit {limit} (max_matrix_size)")
    validate_game_numbers(game, max_scalar_bits=max_scalar_bits, max_scalar_chars=max_scalar_chars)
    return size


def _verify_diagonal(game, diagonal, max_matrix_size):
    from ..analysis.result import AnalysisResult
    constant, weights = xor_coefficients(game, max_matrix_size=max_matrix_size)
    matrix = cost_matrix(weights)
    # Raw inputs are bounded separately; exact algebra may increase coefficient bit sizes.
    verified = _verify_psd_prepared(dual_slack(matrix, diagonal), max_size=max_matrix_size)
    upper = constant + sum(diagonal, Fraction(0))
    payload = {"schema": XOR_DUAL_SCHEMA, "game_fingerprint": game_fingerprint(game),
               "diagonal": list(map(scalar_string, diagonal)), "upper_bound": scalar_string(upper)}
    return AnalysisResult(upper, "quantum_tensor_unbounded_dimension", "exact_xor_dual_verification",
                          "certified_upper_bound", "verified",
                          {"certified": True, "upper_bound": upper, "matrix_size": len(matrix),
                           "psd_rank": verified.rank, "psd_pivots": verified.pivots,
                           "scope": "upper bound on unrestricted bipartite XOR quantum value",
                           "certificate_schema": XOR_DUAL_SCHEMA}, payload)


def verify_xor_dual(game, diagonal, *, max_matrix_size=128,
                    max_scalar_bits=4096, max_scalar_chars=8192):
    """Verify a rational dual diagonal and return its certified quantum upper bound.

    Entries may be Fractions, integers, or rational strings; implicit floating
    reconstruction is forbidden. The value bounds unrestricted finite-dimensional
    and commuting bipartite XOR strategies. No quantum lower bound or optimizer
    is inferred from a numerical Gram matrix. Signed parity rewards are allowed.
    Raw game and diagonal components are bounded before compilation; exact
    derived slack and objective arithmetic may grow beyond those input limits.
    A serialized derived bound may therefore require a larger decoding budget.
    """
    size = _preflight_game(game, max_matrix_size, max_scalar_bits, max_scalar_chars)
    entries = tuple(islice(iter(diagonal), size + 1))
    if len(entries) != size:
        raise ValueError("XOR dual diagonal length does not match the Gram matrix")
    diagonal = tuple(parse_rational(value, max_bits=max_scalar_bits, max_chars=max_scalar_chars)
                     for value in entries)
    return _verify_diagonal(game, diagonal, max_matrix_size)


def verify_xor_certificate(game, certificate, *, max_matrix_size=128,
                           max_scalar_bits=4096, max_scalar_chars=8192):
    """Recheck a saved certificate, its verifier fingerprint, and its claimed bound."""
    fields = {"schema", "game_fingerprint", "diagonal", "upper_bound"}
    if not isinstance(certificate, dict) or set(certificate) != fields:
        raise ValueError("Invalid XOR certificate fields")
    if certificate["schema"] != XOR_DUAL_SCHEMA:
        raise ValueError("Unsupported XOR certificate schema")
    # Check applicability and dimensions before scanning potentially large payloads.
    size = _preflight_game(game, max_matrix_size, max_scalar_bits, max_scalar_chars)
    if not isinstance(certificate["diagonal"], list) or len(certificate["diagonal"]) != size:
        raise ValueError("Serialized XOR diagonal length does not match the Gram matrix")
    if any(not isinstance(value, str) for value in certificate["diagonal"]):
        raise ValueError("Serialized XOR diagonal entries must be rational strings")
    if not isinstance(certificate["upper_bound"], str):
        raise ValueError("Serialized XOR upper bound must be a rational string")
    limits = dict(max_bits=max_scalar_bits, max_chars=max_scalar_chars)
    claimed = parse_rational(certificate["upper_bound"], **limits)
    diagonal = tuple(parse_rational(value, **limits) for value in certificate["diagonal"])
    if certificate["game_fingerprint"] != game_fingerprint(game):
        raise ValueError("XOR certificate is bound to a different verifier")
    result = _verify_diagonal(game, diagonal, max_matrix_size)
    if result.value != claimed:
        raise ValueError("Claimed XOR upper bound differs from the verified dual objective")
    return result
