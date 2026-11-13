"""Portable claim definitions and output receipts; receipts are not proofs by themselves."""
from dataclasses import asdict
from fractions import Fraction

from ..core.identity import canonical_json, verifier_id
from ..core.scalar_wire import parse_rational, scalar_string
from ..core.resource_serialization import resource_from_dict, resource_to_dict
from .inference import claim_id
from .types import ClaimAssessment, GapClaim, ValueClaim

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
