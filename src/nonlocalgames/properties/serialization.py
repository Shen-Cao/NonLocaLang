"""Strict property definitions and output receipts; replay is required for proof."""
from dataclasses import asdict
from fractions import Fraction

from ..core.identity import canonical_json, verifier_id
from ..core.scalar_wire import scalar_string
from ..core.resource_serialization import resource_from_dict, resource_to_dict
from .identity import property_id
from .types import PropertyAssessment, PropertyClaim

PROPERTY_SCHEMA = 'nonlocalgames.property_definition.v1'


def property_claim_to_dict(game, claim):
    if not isinstance(claim, PropertyClaim):
        raise TypeError('Expected PropertyClaim')
    return {'schema': PROPERTY_SCHEMA, 'game_id': verifier_id(game), 'property_id': property_id(game, claim),
            'kind': claim.kind.value, 'resource': None if claim.resource is None else resource_to_dict(claim.resource),
            'off_support': None if claim.off_support is None else claim.off_support.value}


def property_claim_from_dict(game, data, *, max_bytes=8_000_000):
    canonical_json(data, max_bytes=max_bytes)
    fields = {'schema', 'game_id', 'property_id', 'kind', 'resource', 'off_support'}
    if not isinstance(data, dict) or set(data) != fields or data['schema'] != PROPERTY_SCHEMA:
        raise ValueError('Invalid property definition fields or schema')
    if data['game_id'] != verifier_id(game):
        raise ValueError('Property definition belongs to a different verifier')
    resource = None if data['resource'] is None else resource_from_dict(data['resource'])
    claim = PropertyClaim(data['kind'], resource, off_support=data['off_support'])
    if data['property_id'] != property_id(game, claim):
        raise ValueError('Property identity differs from its definition')
    return claim


def _json_value(value):
    if isinstance(value, Fraction):
        return scalar_string(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def property_assessment_to_dict(assessment):
    if not isinstance(assessment, PropertyAssessment):
        raise TypeError('Expected PropertyAssessment')
    result = _json_value(asdict(assessment))
    canonical_json(result)
    return result
