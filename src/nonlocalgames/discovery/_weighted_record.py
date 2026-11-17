"""Raw evidence, replayed gap claims, and numerical screens for one weighted CHSH member."""
from fractions import Fraction
from hashlib import sha256
from math import isfinite

from ..analysis import xor_quantum_value
from ..claims import GapClaim, assess_gap, assessment_to_dict, claim_id, evidence_from_dict, evidence_to_dict
from ..claims.types import EvidenceError, ExecutionStatus
from ..core.identity import JSONBudgetExceeded, canonical_json, verifier_id
from ..core.resources import ResourceSpec
from ..core.scalar_wire import scalar_string
from ..games import biased_chsh
from .weighted_chsh import weighted_chsh_member

_SCHEMA = 'nonlocalgames.weighted_chsh_member.v1'
_CLAIM = GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local'))


def _binding(manifest, k, p, game):
    return {'schema': _SCHEMA, 'run_id': manifest['run_id'], 'k': k, 'p': scalar_string(p),
            'verifier_id': verifier_id(game), 'claim_id': claim_id(game, _CLAIM)}


def _identifier(body):
    return _SCHEMA + ':' + sha256(canonical_json(body).encode()).hexdigest()


def _local_value(result):
    if result is not None and result.execution == 'completed':
        interval = result.intervals[1]
        if interval.lower == interval.upper:
            return interval.lower
    return None


def _screen(game, manifest, local):
    if not manifest['screen']:
        return {'status': 'disabled'}
    try:
        import cvxpy as cp
        if manifest['solver'] not in cp.installed_solvers():
            return {'status': 'unavailable', 'detail': 'Requested solver is unavailable'}
        result = xor_quantum_value(game, solver=manifest['solver'], **manifest['solver_options'])
    except ImportError as error:
        return {'status': 'unavailable', 'detail': str(error)}
    except (cp.error.SolverError, RuntimeError, ValueError) as error:
        return {'status': 'failed', 'detail': str(error)}
    return {'status': 'completed', 'solver': result.diagnostics['solver'],
            'solver_status': result.status, 'value': result.value,
            'candidate_gap': None if local is None else result.value - float(local)}


def _check_screen(screen, manifest, local):
    if not isinstance(screen, dict) or 'status' not in screen:
        raise ValueError('Invalid numerical screen record')
    if not manifest['screen']:
        if screen != {'status': 'disabled'}:
            raise ValueError('Screen differs from the disabled screen policy')
        return
    if screen['status'] in ('unavailable', 'failed'):
        if set(screen) != {'status', 'detail'} or not isinstance(screen['detail'], str):
            raise ValueError('Invalid numerical screen failure record')
        return
    if (set(screen) != {'status', 'solver', 'solver_status', 'value', 'candidate_gap'}
            or screen['status'] != 'completed' or screen['solver'] != manifest['solver']
            or screen['solver_status'] not in ('optimal', 'optimal_inaccurate')
            or type(screen['value']) is not float or not isfinite(screen['value'])):
        raise ValueError('Invalid numerical screen value or solver metadata')
    expected = None if local is None else screen['value'] - float(local)
    if canonical_json(screen['candidate_gap']) != canonical_json(expected):
        raise ValueError('Numerical candidate gap differs from the reported score and exact local value')


def generate_member(manifest, k, budget):
    p = Fraction(k, manifest['steps'])
    game = biased_chsh(p)
    error = None
    try:
        artifacts = weighted_chsh_member(p, grid_denominator=manifest['grid_denominator'],
                                         verification_budget=budget)
    except EvidenceError as exc:
        if exc.issue.code != ExecutionStatus.BUDGET_EXCEEDED:
            raise
        error = {'status': 'budget_exceeded', 'detail': str(exc)}
        artifacts = ()
    result = assess_gap(game, _CLAIM, artifacts, budget=budget) if error is None else None
    if result is not None and result.execution == 'budget_exceeded':
        error = {'status': 'budget_exceeded', 'detail': result.issues[0].message}
        artifacts, result = (), None
    if result is not None and result.execution != 'completed':
        raise ValueError(f'Generated member evidence rejected: {result.execution.value}')
    local = _local_value(result)
    screen = _screen(game, manifest, local)
    _check_screen(screen, manifest, local)
    body = {**_binding(manifest, k, p, game), 'generation_error': error, 'screen': screen,
            'evidence': [evidence_to_dict(a, max_bytes=budget.max_artifact_bytes) for a in artifacts],
            'assessment': None if result is None else assessment_to_dict(result)}
    return {**body, 'member_id': _identifier(body)}


def _review(binding, result, error, screen):
    local = _local_value(result)
    lower, upper = (None, None) if result is None else (result.intervals[0].lower, result.intervals[0].upper)
    gap = None if lower is None or local is None else lower-local
    return {**{key: binding[key] for key in ('k', 'p', 'claim_id')},
            'truth': 'unknown' if result is None else result.truth.value,
            'execution': error['status'] if result is None else result.execution.value,
            'local_value': None if local is None else scalar_string(local),
            'quantum_lower': None if lower is None else scalar_string(lower),
            'quantum_upper': None if upper is None else scalar_string(upper),
            'certified_gap_lower': None if gap is None else scalar_string(gap),
            'screen_status': screen['status'], 'candidate_gap': screen.get('candidate_gap'),
            'generation_error': error}


def verify_member(manifest, k, record, budget):
    """Recompute mathematical claims from raw artifacts. Screen data remain observations."""
    p = Fraction(k, manifest['steps'])
    game = biased_chsh(p)
    binding = _binding(manifest, k, p, game)
    fields = {*binding, 'generation_error', 'evidence', 'assessment', 'screen', 'member_id'}
    if (not isinstance(record, dict) or set(record) != fields
            or canonical_json({key: record[key] for key in binding}) != canonical_json(binding)):
        raise ValueError(f'Invalid weighted CHSH member binding at {k}')
    body = {key: value for key, value in record.items() if key != 'member_id'}
    if record['member_id'] != _identifier(body):
        raise ValueError(f'Weighted CHSH member ID differs from content at {k}')
    raw, error = record['evidence'], record['generation_error']
    if error is not None:
        if (not isinstance(error, dict) or set(error) != {'status', 'detail'}
                or error['status'] != 'budget_exceeded' or not isinstance(error['detail'], str)
                or raw != [] or record['assessment'] is not None):
            raise ValueError('Invalid generation failure record')
        _check_screen(record['screen'], manifest, None)
        return _review(binding, None, error, record['screen'])
    expected = ['local_enumeration', 'quantum_strategy']
    if p <= Fraction(1, 10) or p == 1:
        expected.append('xor_dual')
    if (not isinstance(raw, list) or len(raw) != len(expected)
            or any(not isinstance(item, dict) or item.get('kind') != kind for item, kind in zip(raw, expected))):
        raise ValueError('Unexpected raw evidence list for weighted CHSH member')
    if len(raw) > budget.max_artifacts:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Replay exceeds max_artifacts')
    try:
        artifacts = tuple(evidence_from_dict(item, max_bytes=budget.max_artifact_bytes) for item in raw)
    except JSONBudgetExceeded as exc:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Replay exceeds max_artifact_bytes') from exc
    result = assess_gap(game, _CLAIM, artifacts, budget=budget)
    if result.execution == 'budget_exceeded':
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, f'Replay budget exhausted at member {k}')
    if result.execution != 'completed':
        raise ValueError(f'Archived member {k} has rejected evidence: {result.execution.value}')
    if canonical_json(assessment_to_dict(result)) != canonical_json(record['assessment']):
        raise ValueError(f'Archived claim differs from replay at member {k}')
    _check_screen(record['screen'], manifest, _local_value(result))
    return _review(binding, result, None, record['screen'])
