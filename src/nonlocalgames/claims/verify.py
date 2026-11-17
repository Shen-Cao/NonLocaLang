"""Admission of data artifacts through specific independently replayed checks."""
import json
from math import prod

from ..core.scalar_wire import ScalarBudgetExceeded, parse_rational
from ..core.wire_validation import validate_game_numbers
from ..core.identity import canonical_json, verifier_id
from ..core.resources import Resource, ResourceSpec
from .artifacts import EvidenceArtifact, EvidenceKind
from .types import EvidenceCheck, EvidenceError, ExecutionStatus, VerificationBudget


def _fields(payload, expected):
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError('Unexpected evidence payload fields')


def _quantum_policy(game, budget, identifier):
    try:
        from ..quantum.exact import ExactBudget, ExactBudgetExceeded, validate_game_scalars
    except ImportError as error:
        raise EvidenceError(ExecutionStatus.DEPENDENCY_UNAVAILABLE, str(error), identifier) from error
    policy = ExactBudget(max_joint_dimension=budget.max_quantum_joint_dimension,
                         max_psd_size=budget.max_matrix_size,
                         max_matrix_entries=budget.max_quantum_entries,
                         max_events=budget.max_quantum_events,
                         max_parties=budget.max_quantum_parties,
                         max_scalar_bits=budget.max_quantum_scalar_bits)
    try:
        validate_game_scalars(game, policy)
    except ExactBudgetExceeded as error:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, str(error), identifier) from error
    return policy


def verify_artifact(game, artifact, *, budget=None):
    """Recompute evidence; a supplied AnalysisResult or guarantee label is never trusted."""
    budget = VerificationBudget() if budget is None else budget
    if not isinstance(budget, VerificationBudget):
        raise TypeError('Expected VerificationBudget')
    if not isinstance(artifact, EvidenceArtifact):
        raise EvidenceError(ExecutionStatus.INVALID_EVIDENCE, 'Expected a data EvidenceArtifact, not a result or verified-bound object')
    size = len(artifact.resource_json.encode('utf-8')) + len(artifact.payload_json.encode('utf-8'))
    if size > budget.max_artifact_bytes or artifact.nbytes > budget.max_artifact_bytes:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Evidence exceeds max_artifact_bytes')
    identifier = artifact.artifact_id
    quantum_policy = _quantum_policy(game, budget, identifier) if artifact.kind == EvidenceKind.QUANTUM_STRATEGY else None
    if quantum_policy is None:
        try:
            validate_game_numbers(game, max_scalar_bits=budget.max_wire_scalar_bits,
                                  max_scalar_chars=budget.max_wire_scalar_chars)
        except ScalarBudgetExceeded as error:
            raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, str(error), identifier) from error
    if artifact.game_id != verifier_id(game):
        raise EvidenceError(ExecutionStatus.INVALID_EVIDENCE, 'Evidence belongs to a different verifier', identifier)
    try:
        source = json.loads(artifact.resource_json)
        resource = ResourceSpec(Resource(source['kind']))
        payload = artifact.payload
        if artifact.kind == EvidenceKind.NUMERICAL_HINT:
            if not isinstance(payload, dict):
                raise ValueError('A numerical hint must be a data object')
            return EvidenceCheck(identifier, resource.kind, None, None, 'untrusted_observation.v1', False)
        from ..core.resource_serialization import resource_to_dict
        if artifact.resource_json != canonical_json(resource_to_dict(resource)):
            raise EvidenceError(ExecutionStatus.UNSUPPORTED, 'Proof admission currently supports only standard unrestricted resource models', identifier)
        if artifact.kind in (EvidenceKind.LOCAL_STRATEGY, EvidenceKind.LOCAL_ENUMERATION):
            if resource.kind != Resource.LOCAL:
                raise EvidenceError(ExecutionStatus.UNSUPPORTED, 'Local evidence must declare the standard local source model', identifier)
            if artifact.kind == EvidenceKind.LOCAL_ENUMERATION:
                counts = tuple(a**x for x, a in zip(game.scenario.input_sizes, game.scenario.output_sizes))
                if prod(counts)//max(counts) > budget.max_strategies:
                    raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Classical replay exceeds max_strategies', identifier)
            from ..classical import DeterministicStrategy
            _fields(payload, {'strategy'} if artifact.kind == EvidenceKind.LOCAL_STRATEGY else {'strategy', 'value'})
            claimed_value = (parse_rational(payload['value'], max_bits=budget.max_wire_scalar_bits,
                                            max_chars=budget.max_wire_scalar_chars)
                             if artifact.kind == EvidenceKind.LOCAL_ENUMERATION else None)
            witness = DeterministicStrategy(game.scenario, payload['strategy'])
            lower = witness.score(game)
            if artifact.kind == EvidenceKind.LOCAL_STRATEGY:
                return EvidenceCheck(identifier, resource.kind, lower, None, 'deterministic_score.v1', attained_score=lower)
            from ..analysis.classical import classical_value
            exact = classical_value(game, max_strategies=budget.max_strategies)
            if lower != exact.value or claimed_value != exact.value:
                raise ValueError('Local enumeration payload does not match independent exact replay')
            return EvidenceCheck(identifier, resource.kind, exact.value, exact.value, 'local_enumeration_replay.v1', attained_score=lower)
        if artifact.kind == EvidenceKind.NONSIGNALLING_LP:
            if resource.kind != Resource.NONSIGNALLING:
                raise EvidenceError(ExecutionStatus.UNSUPPORTED, 'NS certificates must declare the standard nonsignalling source model', identifier)
            _fields(payload, {'primal', 'dual'})
            if game.scenario.entries > budget.max_variables:
                raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'NS verification exceeds max_variables', identifier)
            for field in ('primal', 'dual'):
                vector = payload[field]
                if not isinstance(vector, list) or any(not isinstance(entry, str) for entry in vector):
                    raise ValueError(f'NS certificate {field} must be a JSON array of rational strings')
            from ..analysis.certificates import verify_nonsignalling_certificate
            result = verify_nonsignalling_certificate(
                game, payload['primal'], payload['dual'], max_variables=budget.max_variables,
                max_scalar_bits=budget.max_wire_scalar_bits,
                max_scalar_chars=budget.max_wire_scalar_chars)
            return EvidenceCheck(identifier, resource.kind, result.diagnostics['lower_bound'],
                                  result.diagnostics['upper_bound'], 'rational_ns_lp.v1',
                                  attained_score=result.diagnostics['lower_bound'])
        if artifact.kind == EvidenceKind.QUANTUM_STRATEGY:
            if resource.kind != Resource.QUANTUM_FINITE:
                raise EvidenceError(ExecutionStatus.UNSUPPORTED, 'Quantum strategy certificates use the standard finite tensor-product source model', identifier)
            from ..quantum.exact import ExactBudgetExceeded
            from ..certificates.quantum import verify_quantum_lower_certificate
            try:
                result = verify_quantum_lower_certificate(game, payload, budget=quantum_policy,
                                                          max_bytes=budget.max_artifact_bytes)
            except ExactBudgetExceeded as error:
                raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, str(error), identifier) from error
            return EvidenceCheck(identifier, resource.kind, result.value, None, 'rational_quantum_strategy.v1',
                                 attained_score=result.value, actual_dimensions=tuple(result.diagnostics['actual_local_dimensions']))
        if artifact.kind == EvidenceKind.XOR_PRIMAL:
            if resource.kind != Resource.QUANTUM_FINITE:
                raise EvidenceError(ExecutionStatus.UNSUPPORTED,
                                    'XOR primal certificates require the finite tensor-product model', identifier)
            if sum(game.scenario.input_sizes) > budget.max_matrix_size:
                raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'XOR verification exceeds max_matrix_size', identifier)
            from ..certificates.xor_primal import verify_xor_primal_certificate
            result = verify_xor_primal_certificate(
                game, payload, max_matrix_size=budget.max_matrix_size,
                max_scalar_bits=budget.max_wire_scalar_bits, max_scalar_chars=budget.max_wire_scalar_chars)
            return EvidenceCheck(identifier, resource.kind, result.value, None, 'rational_xor_primal_clifford.v1',
                                 attained_score=result.value, actual_dimensions=result.diagnostics['realization_dimensions'])
        if artifact.kind == EvidenceKind.XOR_DUAL:
            if resource.kind != Resource.QUANTUM_COMMUTING:
                raise EvidenceError(ExecutionStatus.UNSUPPORTED, 'XOR duals use the standard commuting upper-bound source model', identifier)
            if sum(game.scenario.input_sizes) > budget.max_matrix_size:
                raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'XOR verification exceeds max_matrix_size', identifier)
            from ..certificates.xor import verify_xor_certificate
            result = verify_xor_certificate(
                game, payload, max_matrix_size=budget.max_matrix_size,
                max_scalar_bits=budget.max_wire_scalar_bits,
                max_scalar_chars=budget.max_wire_scalar_chars)
            return EvidenceCheck(identifier, resource.kind, None, result.value, 'rational_xor_dual.v1')
        raise EvidenceError(ExecutionStatus.UNSUPPORTED, 'No verifier is registered for this evidence kind', identifier)
    except EvidenceError:
        raise
    except ScalarBudgetExceeded as error:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, str(error), identifier) from error
    except ImportError as error:
        raise EvidenceError(ExecutionStatus.DEPENDENCY_UNAVAILABLE, str(error), identifier) from error
    except RuntimeError as error:
        raise EvidenceError(ExecutionStatus.CHECKER_ERROR, str(error), identifier) from error
    except (TypeError, ValueError, KeyError, OverflowError, ZeroDivisionError) as error:
        raise EvidenceError(ExecutionStatus.INVALID_EVIDENCE, str(error), identifier) from error
