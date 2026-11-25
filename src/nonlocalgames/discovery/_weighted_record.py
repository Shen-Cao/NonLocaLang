"""Raw evidence, replayed gap claims, and numerical screens for one weighted CHSH member."""
from fractions import Fraction
from hashlib import sha256
from math import isfinite, prod

from ..analysis import xor_quantum_value
from ..certificates.quantum import QUANTUM_LOWER_SCHEMA
from ..certificates.xor import XOR_DUAL_SCHEMA
from ..certificates.xor_model import xor_coefficients
from ..claims import (EvidenceKind, GapClaim, VerificationBudget, assess_gap, assessment_to_dict, claim_id,
                      evidence_from_dict, evidence_to_dict)
from ..claims.admission import bound_intervals
from ..claims.serialization import assessment_from_dict
from ..claims.types import ClaimAssessment, EvidenceError, ExecutionStatus, TruthStatus
from ..claims.verify import _quantum_policy
from ..classical.strategy import response_tables
from ..core.identity import JSONBudgetExceeded, VERIFIER_ID_PREFIX, canonical_json, verifier_id
from ..core.resources import Resource, ResourceSpec
from ..core.scalar_wire import parse_rational, scalar_string
from ..games import biased_chsh
from ..quantum.exact.serialization import STRATEGY_SCHEMA, matrix_from_dict
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


def prepare_template(manifest):
    """Return an invocation-local screening method or a recorded initialization failure."""
    if manifest.get('template_policy') != 'fresh_solver_state':
        raise ValueError('Expected the fresh_solver_state template policy')
    try:
        import cvxpy as cp
    except ImportError as error:
        return None, {'status': 'unavailable', 'detail': str(error)}
    try:
        if manifest['solver'] not in cp.installed_solvers():
            return None, {'status': 'unavailable', 'detail': 'Requested solver is unavailable'}
        from ..analysis.xor_template import XorScreenSession
        session = XorScreenSession(biased_chsh(0).scenario, solver=manifest['solver'],
                                   **manifest['solver_options'])
        return session.solve, None
    except ImportError as error:
        return None, {'status': 'unavailable', 'detail': str(error)}
    except (cp.error.SolverError, RuntimeError, ValueError) as error:
        return None, {'status': 'failed', 'detail': str(error)}


def _screen(game, manifest, local, template=None):
    if not manifest['screen']:
        return {'status': 'disabled'}
    if template is not None and template[1] is not None:
        return template[1].copy()
    try:
        import cvxpy as cp
        if template is not None:
            solve, _ = template
            result = solve(game)
        else:
            if 'template_policy' in manifest:
                raise RuntimeError('Template run has no screening session')
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


def check_screen_shape(screen, manifest):
    """Check archived screen fields without making a claim about the observed score."""
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
    candidate = screen['candidate_gap']
    if candidate is not None and (type(candidate) is not float or not isfinite(candidate)):
        raise ValueError('Invalid numerical candidate gap')


def _check_screen(screen, manifest, local):
    check_screen_shape(screen, manifest)
    if screen['status'] != 'completed':
        return
    expected = None if local is None else screen['value'] - float(local)
    if canonical_json(screen['candidate_gap']) != canonical_json(expected):
        raise ValueError('Numerical candidate gap differs from the reported score and exact local value')


def generate_member(manifest, k, budget, *, template=None):
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
    screen = _screen(game, manifest, local, template)
    _check_screen(screen, manifest, local)
    body = {**_binding(manifest, k, p, game), 'generation_error': error, 'screen': screen,
            'evidence': [evidence_to_dict(a, max_bytes=budget.max_artifact_bytes) for a in artifacts],
            'assessment': None if result is None else assessment_to_dict(result)}
    return {**body, 'member_id': _identifier(body)}


def _review(binding, result, error, screen):
    local = _local_value(result)
    lower, upper = None, None
    if result is not None and result.execution == ExecutionStatus.COMPLETED:
        lower, upper = result.intervals[0].lower, result.intervals[0].upper
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


def validate_member_binding(manifest, k, record):
    """Check the manifest, grid, claim and content bindings without replaying evidence."""
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
    else:
        expected = ['local_enumeration', 'quantum_strategy']
        if p <= Fraction(1, 10) or p == 1:
            expected.append('xor_dual')
        if (not isinstance(raw, list) or len(raw) != len(expected)
                or any(not isinstance(item, dict) or item.get('kind') != kind for item, kind in zip(raw, expected))):
            raise ValueError('Unexpected raw evidence list for weighted CHSH member')
    return game, binding


def _weighted_payload(artifact, fields, *, schema=None):
    payload = artifact.payload
    if not isinstance(payload, dict) or payload.keys() != fields:
        raise ValueError(f'Invalid {artifact.kind.value} payload fields')
    if schema is not None and (payload['schema'] != schema
                               or payload['game_fingerprint'] != artifact.game_id.removeprefix(VERIFIER_ID_PREFIX)):
        raise ValueError(f'Invalid {artifact.kind.value} payload schema or game fingerprint')
    return payload


def admission_budget(manifest):
    """Use the validated manifest's limits to read its archived observations."""
    return VerificationBudget(**manifest['verification_budget'])


def _payload_rational(value, field, *, budget):
    if not isinstance(value, str):
        raise ValueError(f'Weighted payload {field} must be a canonical rational string')
    return parse_rational(value, canonical=True, max_bits=budget.max_wire_scalar_bits,
                          max_chars=budget.max_wire_scalar_chars)


def _validate_quantum_strategy(strategy, scenario, budget):
    """Parse the state and effects under the archive limits. Feasibility is checked in replay."""
    fields = {'schema', 'encoding', 'dimensions', 'state', 'measurements'}
    if not isinstance(strategy, dict) or strategy.keys() != fields or strategy['schema'] != STRATEGY_SCHEMA:
        raise ValueError('Invalid exact quantum strategy fields or schema')
    if strategy['encoding'] != 'rational_complex':
        raise ValueError('Exact quantum strategies require rational_complex encoding')
    dimensions = strategy['dimensions']
    if (not isinstance(dimensions, list) or len(dimensions) != scenario.parties
            or any(type(d) is not int or d <= 0 for d in dimensions)):
        raise ValueError('Weighted quantum_strategy payload dimensions must be two positive integers')

    joint = prod(dimensions)
    if joint > budget.max_joint_dimension:
        raise ValueError('Quantum strategy exceeds max_joint_dimension')
    parties = tuple(zip(dimensions, scenario.input_sizes, scenario.output_sizes))
    entries = joint**2 + sum(inputs * outputs * d**2 for d, inputs, outputs in parties)
    if entries > budget.max_matrix_entries:
        raise ValueError('Quantum state and effects exceed max_matrix_entries')
    measurements = strategy['measurements']
    if not isinstance(measurements, list) or len(measurements) != scenario.parties:
        raise ValueError('Quantum strategy requires measurements for each party')
    matrix_from_dict(strategy['state'], joint, budget=budget)
    for party, (dimension, inputs, outputs) in zip(measurements, parties):
        if not isinstance(party, list) or len(party) != inputs:
            raise ValueError('Quantum strategy requires one measurement per local input')
        for povm in party:
            if not isinstance(povm, list) or len(povm) != outputs:
                raise ValueError('Quantum measurement requires one effect per local output')
            for effect in povm:
                matrix_from_dict(effect, dimension, budget=budget)


def _validate_weighted_check(check, artifact, *, game, budget):
    """Validate payload structure and archived check shape. Replay compares their results."""
    if artifact.kind == EvidenceKind.LOCAL_ENUMERATION:
        checker, resource = 'local_enumeration_replay.v1', Resource.LOCAL
        payload = _weighted_payload(artifact, {'strategy', 'value'})
        _payload_rational(payload['value'], 'value', budget=budget)
        tables = payload['strategy']
        if (not isinstance(tables, list) or len(tables) != game.scenario.parties
                or any(not isinstance(t, list) or len(t) != n for t, n in zip(tables, game.scenario.input_sizes))):
            raise ValueError('Local strategy requires response tables with one output per local input')
        response_tables(game.scenario, tables)
        valid_bounds = (check.lower is not None and check.lower == check.upper == check.attained_score
                        and check.actual_dimensions is None)
    elif artifact.kind == EvidenceKind.QUANTUM_STRATEGY:
        checker, resource = 'rational_quantum_strategy.v1', Resource.QUANTUM_FINITE
        payload = _weighted_payload(artifact, {'schema', 'resource', 'game_fingerprint', 'strategy', 'lower_bound'},
                                    schema=QUANTUM_LOWER_SCHEMA)
        if payload['resource'] != resource.value:
            raise ValueError('Invalid quantum_strategy payload resource')
        _payload_rational(payload['lower_bound'], 'lower_bound', budget=budget)
        policy = _quantum_policy(game, budget, check.artifact_id)
        _validate_quantum_strategy(payload['strategy'], game.scenario, policy)
        valid_bounds = (check.lower is not None and check.lower == check.attained_score and check.upper is None
                        and check.actual_dimensions is not None and len(check.actual_dimensions) == game.scenario.parties)
    elif artifact.kind == EvidenceKind.XOR_DUAL:
        checker, resource = 'rational_xor_dual.v1', Resource.QUANTUM_COMMUTING
        payload = _weighted_payload(artifact, {'schema', 'game_fingerprint', 'diagonal', 'upper_bound'},
                                    schema=XOR_DUAL_SCHEMA)
        diagonal = payload['diagonal']
        # The weighted CHSH Gram matrix has two inputs for each of two parties.
        if not isinstance(diagonal, list) or len(diagonal) != 4:
            raise ValueError('Weighted xor_dual payload diagonal must have four entries')
        constant, _ = xor_coefficients(game, max_matrix_size=budget.max_matrix_size)
        objective = sum((_payload_rational(entry, 'diagonal', budget=budget) for entry in diagonal), constant)
        if objective != _payload_rational(payload['upper_bound'], 'upper_bound', budget=budget):
            raise ValueError('XOR diagonal objective differs from its declared upper bound')
        valid_bounds = (check.lower is None and check.upper is not None
                        and check.attained_score is None and check.actual_dimensions is None)
    else:
        raise ValueError(f'Unexpected weighted evidence kind {artifact.kind.value}')
    if (not check.is_proof or check.checker != checker or check.resource != resource
            or artifact.resource != ResourceSpec(resource)):
        raise ValueError(f'Weighted {artifact.kind.value} evidence requires a {checker} '
                         f'proof check on {resource.value}')
    if not valid_bounds:
        raise ValueError(f'Invalid bounds or dimensions for the {checker} check')


def validate_observed_assessment(record, artifacts, *, game, budget):
    """Check the internal consistency of a weighted receipt. Proof still requires replay."""
    artifacts = tuple(artifacts)
    assessment = assessment_from_dict(record['assessment'], claim_id=record['claim_id'],
                                      game_id=record['verifier_id'], artifacts=artifacts,
                                      max_scalar_bits=budget.max_wire_scalar_bits,
                                      max_scalar_chars=budget.max_wire_scalar_chars)
    if assessment.execution != ExecutionStatus.COMPLETED:
        raise ValueError('Expected a completed weighted receipt')
    if assessment.numerical_hints or assessment.issues:
        raise ValueError('A completed weighted receipt must have empty numerical_hints and issues')

    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    if (len(assessment.verified_evidence) != len(by_id)
            or set(assessment.verified_evidence) != by_id.keys()):
        raise ValueError('Weighted verified_evidence must name each member artifact exactly once')
    if len(assessment.checks) != len(artifacts):
        raise ValueError('Weighted checks must cover each member artifact exactly once')
    for check in assessment.checks:
        artifact = by_id.pop(check.artifact_id, None)
        if artifact is None:
            raise ValueError(f'Duplicate weighted check for artifact {check.artifact_id}')
        _validate_weighted_check(check, artifact, game=game, budget=budget)

    intervals, rules = bound_intervals(assessment.checks)
    by_resource = {interval.resource: interval for interval in intervals}
    expected = (by_resource[_CLAIM.higher.kind], by_resource[_CLAIM.lower.kind])
    if assessment.intervals != expected:
        raise ValueError('Weighted receipt intervals must equal the derived quantum_finite and local intervals')
    if assessment.derivation != rules:
        raise ValueError('Weighted receipt derivation differs from the rules for its checks')
    if any(i.lower is not None and i.upper is not None and i.lower > i.upper for i in intervals):
        raise ValueError('Inconsistent bounds in completed weighted receipt intervals')
    high, low = expected
    truth = TruthStatus.UNKNOWN
    if high.lower is not None and low.upper is not None and high.lower - low.upper > _CLAIM.threshold:
        truth = TruthStatus.PROVED
    elif high.upper is not None and low.lower is not None and high.upper - low.lower <= _CLAIM.threshold:
        truth = TruthStatus.DISPROVED
    if assessment.truth != truth:
        raise ValueError('Archived truth differs from the strict gap verdict of its derived intervals')


def verify_member_assessment(manifest, k, record, budget, *,
                             preserve_incomplete=False) -> tuple[ClaimAssessment | None, dict]:
    """Return the replayed assessment and review. A generation failure has only a review.

    Ledger callers can preserve a returned execution failure and its issues. Exceptions
    raised while decoding or replaying evidence continue to propagate.
    """
    game, binding = validate_member_binding(manifest, k, record)
    raw, error = record['evidence'], record['generation_error']
    if error is not None:
        return None, _review(binding, None, error, record['screen'])
    if len(raw) > budget.max_artifacts:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Replay exceeds max_artifacts')
    try:
        artifacts = tuple(evidence_from_dict(item, max_bytes=budget.max_artifact_bytes) for item in raw)
    except JSONBudgetExceeded as exc:
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, 'Replay exceeds max_artifact_bytes') from exc
    if preserve_incomplete:
        # A returned failure cannot reproduce the archived verdict. Validate its
        # observation before replay so even a failed attempt has a bound origin.
        check_screen_shape(record['screen'], manifest)
        validate_observed_assessment(record, artifacts, game=game, budget=admission_budget(manifest))
    result = assess_gap(game, _CLAIM, artifacts, budget=budget)
    if preserve_incomplete and result.execution != ExecutionStatus.COMPLETED:
        return result, _review(binding, result, None, record['screen'])
    if result.execution == 'budget_exceeded':
        raise EvidenceError(ExecutionStatus.BUDGET_EXCEEDED, f'Replay budget exhausted at member {k}')
    if result.execution != 'completed':
        raise ValueError(f'Archived member {k} has rejected evidence: {result.execution.value}')
    if canonical_json(assessment_to_dict(result)) != canonical_json(record['assessment']):
        raise ValueError(f'Archived claim differs from replay at member {k}')
    _check_screen(record['screen'], manifest, _local_value(result))
    return result, _review(binding, result, None, record['screen'])


def verify_member(manifest, k, record, budget):
    """Recompute mathematical claims from raw artifacts. Screen data remain observations."""
    _, review = verify_member_assessment(manifest, k, record, budget)
    return review
