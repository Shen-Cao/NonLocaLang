"""Attained finite quantum scores from exact bipartite XOR correlation Gram matrices."""
from fractions import Fraction

from ..core.scalar_wire import parse_rational, scalar_string
from .linalg import _bounded_rows, _verify_psd_prepared
from .xor import _preflight_game
from .xor_model import game_fingerprint, xor_coefficients


XOR_PRIMAL_SCHEMA = 'nonlocalgames.xor_primal.v1'


def _check(game, gram, fingerprint, max_matrix_size):
    from ..analysis.result import AnalysisResult
    constant, weights = xor_coefficients(game, max_matrix_size=max_matrix_size)
    if any(row[i] != 1 for i, row in enumerate(gram)):
        raise ValueError('XOR primal Gram diagonal must equal one')
    checked = _verify_psd_prepared(gram, max_size=max_matrix_size)
    alice_inputs = len(weights)
    score = constant + sum((w * gram[x][alice_inputs+y]
                            for x, row in enumerate(weights) for y, w in enumerate(row)), Fraction(0))
    dimension = 1 << (checked.rank // 2)
    payload = {'schema': XOR_PRIMAL_SCHEMA, 'game_fingerprint': fingerprint,
               'gram': [list(map(scalar_string, row)) for row in gram], 'achieved_score': scalar_string(score)}
    return AnalysisResult(score, 'quantum_finite', 'exact_xor_primal_verification',
                          'certified_lower_bound', 'verified',
                          {'certified': True, 'achieved_score': score, 'gram_rank': checked.rank,
                           'realization_dimensions': (dimension, dimension),
                           'scope': 'attained bipartite XOR score by finite Clifford realization',
                           'certificate_schema': XOR_PRIMAL_SCHEMA}, payload)


def verify_xor_gram(game, gram, *, max_matrix_size=128, max_scalar_bits=4096, max_scalar_chars=8192):
    """Check a real PSD unit-diagonal Gram matrix in Alice-then-Bob question order.

    For rank r, choose real unit factors and r pairwise anticommuting Hermitian
    Clifford generators on dimension 2^floor(r/2). Linear combinations of these
    generators with the unit vectors give dichotomic observables. Transposing
    Bob's observables and using a maximally entangled state realizes the cross
    block as correlators. Rank one uses a single scalar generator. Diagnostics
    give the dimensions of this construction; minimal dimensions can be smaller.
    The checker performs exact elimination on the supplied Gram matrix alone.
    """
    size = _preflight_game(game, max_matrix_size, max_scalar_bits, max_scalar_chars)
    rows = _bounded_rows(gram, size)
    if len(rows) != size:
        raise ValueError('XOR primal Gram size differs from the question counts')
    prepared = tuple(tuple(parse_rational(v, max_bits=max_scalar_bits, max_chars=max_scalar_chars)
                           for v in row) for row in rows)
    return _check(game, prepared, game_fingerprint(game), max_matrix_size)


def verify_xor_primal_certificate(game, certificate, *, max_matrix_size=128, max_scalar_bits=4096,
                                  max_scalar_chars=8192):
    """Recompute a saved primal score and check its exact verifier binding."""
    fields = {'schema', 'game_fingerprint', 'gram', 'achieved_score'}
    if not isinstance(certificate, dict) or set(certificate) != fields or certificate['schema'] != XOR_PRIMAL_SCHEMA:
        raise ValueError('Invalid XOR primal certificate fields or schema')
    size = _preflight_game(game, max_matrix_size, max_scalar_bits, max_scalar_chars)
    rows = certificate['gram']
    if not isinstance(rows, list) or len(rows) != size or any(
            not isinstance(row, list) or len(row) != size for row in rows):
        raise ValueError('Serialized XOR primal Gram must be a square array of the required size')
    if not isinstance(certificate['achieved_score'], str) or any(
            not isinstance(entry, str) for row in rows for entry in row):
        raise ValueError('Serialized XOR primal scores and entries must be canonical rational strings')
    limits = dict(max_bits=max_scalar_bits, max_chars=max_scalar_chars, canonical=True)
    claimed = parse_rational(certificate['achieved_score'], **limits)
    gram = tuple(tuple(parse_rational(v, **limits) for v in row) for row in rows)
    fingerprint = game_fingerprint(game)
    if certificate['game_fingerprint'] != fingerprint:
        raise ValueError('XOR primal certificate is bound to a different verifier')
    result = _check(game, gram, fingerprint, max_matrix_size)
    if claimed != result.value:
        raise ValueError('Claimed XOR primal score differs from the verified Gram objective')
    return result
