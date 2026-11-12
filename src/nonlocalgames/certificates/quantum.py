"""Exact finite-dimensional strategy scores as certified quantum lower bounds."""
from dataclasses import replace

from ..core.identity import canonical_json, verifier_digest

QUANTUM_LOWER_SCHEMA = 'nonlocalgames.quantum_lower.v1'


def verify_quantum_lower_certificate(game, certificate, *, budget=None, max_bytes=8_000_000):
    """Recheck exact density/POVM data and Born score without numerical feasibility tests.

    This proves an achievable score in the unrestricted finite tensor-product
    model. It asserts neither global optimality nor preservation of a fixed state,
    PPT, projective, product-state, or independent-source contract.
    """
    from ..analysis.result import AnalysisResult
    from ..quantum.exact import ExactBudget, strategy_from_dict, validate_game_scalars
    budget = ExactBudget() if budget is None else budget
    validate_game_scalars(game, budget)
    canonical_json(certificate, max_bytes=max_bytes)
    fields = {'schema', 'resource', 'game_fingerprint', 'strategy', 'lower_bound'}
    if not isinstance(certificate, dict) or set(certificate) != fields:
        raise ValueError('Invalid quantum lower-certificate fields')
    if certificate['schema'] != QUANTUM_LOWER_SCHEMA or certificate['resource'] != 'quantum_finite':
        raise ValueError('Quantum lower certificate must use the finite tensor-product schema')
    if certificate['game_fingerprint'] != verifier_digest(game):
        raise ValueError('Quantum lower certificate belongs to a different verifier')
    if not isinstance(certificate['lower_bound'], str):
        raise ValueError('The certified lower score must be a rational string')
    from ..quantum.exact.scalars import exact_scalar, scalar_string
    claimed = exact_scalar(certificate['lower_bound'], max_bits=budget.max_scalar_bits)
    strategy = strategy_from_dict(certificate['strategy'], budget=budget)
    value = strategy.score(game, budget=budget)
    if claimed != value:
        raise ValueError('Claimed quantum lower score differs from the exact Born score')
    from ..quantum.exact import strategy_to_dict
    checked = {'schema': QUANTUM_LOWER_SCHEMA, 'resource': 'quantum_finite',
               'game_fingerprint': verifier_digest(game),
               'strategy': strategy_to_dict(strategy), 'lower_bound': scalar_string(value)}
    return AnalysisResult(value, 'quantum_finite', 'exact_quantum_strategy_verification',
        'certified_lower_bound', 'verified',
        {'certified': True, 'actual_local_dimensions': strategy.dimensions,
         'scope': 'achievable finite tensor-product POVM score; not global optimality',
         'certificate_schema': QUANTUM_LOWER_SCHEMA,
         'physical_dimension_is_not_realification_size': True}, checked)


def certify_quantum_strategy(game, strategy, *, budget=None, max_bytes=8_000_000):
    """Package an exact strategy, then independently re-decode and verify its score."""
    from ..quantum.exact import (ExactBudget, ExactQuantumStrategy, strategy_to_dict,
                                 validate_game_scalars)
    if not isinstance(strategy, ExactQuantumStrategy):
        raise TypeError('Expected ExactQuantumStrategy; numerical candidates need explicit rationalization')
    budget = ExactBudget() if budget is None else budget
    validate_game_scalars(game, budget)
    value = strategy.score(game, budget=budget)
    from ..quantum.exact.scalars import scalar_string
    certificate = {'schema': QUANTUM_LOWER_SCHEMA, 'resource': 'quantum_finite',
                   'game_fingerprint': verifier_digest(game),
                   'strategy': strategy_to_dict(strategy), 'lower_bound': scalar_string(value)}
    return verify_quantum_lower_certificate(game, certificate, budget=budget, max_bytes=max_bytes)


def certify_quantum_lower_bound(game, strategy, *, max_denominator=1_000_000,
                                repair_steps=30, budget=None, max_bytes=8_000_000):
    """Discover an exact feasible approximation, then verify its achieved score.

    Rounding and repair change the candidate. Only the final exact realization
    is certified, under the unrestricted finite tensor-product resource model.
    """
    from ..quantum.exact import ExactBudget, validate_game_scalars
    from ..quantum.exact.rounding import rationalize_strategy
    budget = ExactBudget() if budget is None else budget
    validate_game_scalars(game, budget)
    repaired = rationalize_strategy(strategy, max_denominator=max_denominator,
                                    repair_steps=repair_steps, budget=budget)
    checked = certify_quantum_strategy(game, repaired.strategy, budget=budget, max_bytes=max_bytes)
    return replace(checked, method='rational_quantum_lower_certificate', diagnostics={
        **checked.diagnostics, 'rationalization': repaired.diagnostics,
        'source_restrictions_preserved': 'physical dimensions only; no other restriction is asserted'})
