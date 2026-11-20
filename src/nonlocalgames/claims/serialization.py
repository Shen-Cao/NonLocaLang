"""Portable claim definitions and output receipts; receipts are not proofs by themselves."""
from dataclasses import asdict
from fractions import Fraction

from ..core.identity import canonical_json, verifier_id
from ..core.scalar_wire import parse_rational, scalar_string
from ..core.resource_serialization import resource_from_dict, resource_to_dict
from ..core.resources import Resource
from .artifacts import EvidenceArtifact
from .inference import claim_id
from .types import (CertifiedInterval, ClaimAssessment, ClaimIssue, EvidenceCheck, ExecutionStatus,
                    GapClaim, ValueClaim)

SCHEMA = 'nonlocalgames.claim_definition.v1'


def claim_to_dict(game, claim):
    result = {'schema': SCHEMA, 'game_id': verifier_id(game), 'claim_id': claim_id(game, claim),
              'threshold': scalar_string(claim.threshold)}
    if isinstance(claim, ValueClaim):
        result.update(kind='value', resource=resource_to_dict(claim.resource), relation=claim.relation.value)
    elif isinstance(claim, GapClaim):
        result.update(kind='gap_greater', higher=resource_to_dict(claim.higher), lower=resource_to_dict(claim.lower))
    else:
        raise TypeError('Expected ValueClaim or GapClaim')
    return result


def claim_from_dict(game, data, *, max_bytes=8_000_000, max_scalar_bits=4096, max_scalar_chars=8192):
    canonical_json(data, max_bytes=max_bytes)
    base = {'schema', 'game_id', 'claim_id', 'threshold', 'kind'}
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise ValueError('Invalid claim definition schema')
    if data.get('game_id') != verifier_id(game):
        raise ValueError('Claim definition belongs to a different verifier')
    if not isinstance(data.get('threshold'), str):
        raise ValueError('Serialized claim thresholds must be rational strings')
    threshold = parse_rational(data['threshold'], max_bits=max_scalar_bits, max_chars=max_scalar_chars)
    if data.get('kind') == 'value' and set(data) == base | {'resource', 'relation'}:
        claim = ValueClaim(resource_from_dict(data['resource']), data['relation'], threshold)
    elif data.get('kind') == 'gap_greater' and set(data) == base | {'higher', 'lower'}:
        claim = GapClaim(resource_from_dict(data['higher']), resource_from_dict(data['lower']), threshold)
    else:
        raise ValueError('Unexpected claim fields or kind')
    if data['claim_id'] != claim_id(game, claim):
        raise ValueError('Claim identity differs from its definition')
    return claim


def _json_value(value):
    if isinstance(value, Fraction):
        return scalar_string(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def assessment_to_dict(assessment):
    """Encode a receipt; replay the archived claim and raw artifacts to check it."""
    if not isinstance(assessment, ClaimAssessment):
        raise TypeError('Expected ClaimAssessment')
    result = _json_value(asdict(assessment))
    canonical_json(result)
    return result


def _receipt_fields(data, record_type):
    if not isinstance(data, dict) or data.keys() != record_type.__dataclass_fields__.keys():
        raise ValueError(f'Invalid archived {record_type.__name__} fields')


def _receipt_array(data, field):
    value = data[field]
    if not isinstance(value, list):
        raise ValueError(f'Archived {field} must be a JSON array')
    return value


def _receipt_text(value, field, *, allow_empty=False):
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f'Invalid archived string for {field}')
    return value


def assessment_from_dict(data, *, claim_id, game_id, artifacts,
                         max_scalar_bits=4096, max_scalar_chars=8192) -> ClaimAssessment:
    """Decode a standard v2 receipt bound to the given claim and raw evidence envelopes.

    This checks the wire representation and reference ownership. All decoded fields,
    including the verdict and checker outputs, are archived observations. Admission
    of proof requires an independent replay of the artifacts.
    """
    _receipt_fields(data, ClaimAssessment)
    if (data['schema'] != ClaimAssessment.schema or _receipt_text(data['claim_id'], 'claim_id') != claim_id
            or _receipt_text(data['game_id'], 'game_id') != game_id):
        raise ValueError('Invalid archived assessment schema or claim/game binding')
    ids = set()
    for artifact in artifacts:
        if not isinstance(artifact, EvidenceArtifact) or artifact.game_id != game_id:
            raise ValueError('Archived evidence belongs to a different game verifier')
        ids.add(artifact.artifact_id)

    def reference(value):
        if not isinstance(value, str) or value not in ids:
            raise ValueError('Archived evidence reference is outside the supplied envelopes')
        return value

    def references(record, field):
        return tuple(reference(value) for value in _receipt_array(record, field))

    def scalar(value):
        if value is None:
            return None
        return parse_rational(_receipt_text(value, 'rational'), canonical=True,
                              max_bits=max_scalar_bits, max_chars=max_scalar_chars)

    intervals = []
    for item in _receipt_array(data, 'intervals'):
        _receipt_fields(item, CertifiedInterval)
        intervals.append(CertifiedInterval(
            Resource(item['resource']), scalar(item['lower']), scalar(item['upper']),
            references(item, 'lower_refs'), references(item, 'upper_refs')))
    issues = []
    for item in _receipt_array(data, 'issues'):
        _receipt_fields(item, ClaimIssue)
        evidence_id = item['evidence_id']
        message = _receipt_text(item['message'], 'message', allow_empty=True)
        issues.append(ClaimIssue(ExecutionStatus(item['code']), message,
                                 None if evidence_id is None else reference(evidence_id)))
    checks = []
    for item in _receipt_array(data, 'checks'):
        _receipt_fields(item, EvidenceCheck)
        dimensions = item['actual_dimensions']
        checks.append(EvidenceCheck(
            reference(item['artifact_id']), Resource(item['resource']), scalar(item['lower']),
            scalar(item['upper']), _receipt_text(item['checker'], 'checker'), item['is_proof'],
            scalar(item['attained_score']),
            None if dimensions is None else tuple(_receipt_array(item, 'actual_dimensions'))))
    return ClaimAssessment(
        claim_id=claim_id, game_id=game_id, truth=data['truth'], execution=data['execution'],
        intervals=tuple(intervals), verified_evidence=references(data, 'verified_evidence'),
        numerical_hints=references(data, 'numerical_hints'), issues=tuple(issues), checks=tuple(checks),
        derivation=tuple(_receipt_text(rule, 'derivation ID') for rule in _receipt_array(data, 'derivation')),
    )
