"""Property identities bind their precise meaning and verifier representation."""
from hashlib import sha256

from ..core.identity import canonical_json, resource_id, verifier_id
from .types import PropertyClaim


def property_id(game, claim):
    if not isinstance(claim, PropertyClaim):
        raise TypeError('Expected PropertyClaim')
    data = {'game_id': verifier_id(game), 'kind': claim.kind.value,
            'resource_id': None if claim.resource is None else resource_id(claim.resource),
            'off_support': None if claim.off_support is None else claim.off_support.value}
    return 'nonlocalgames.property.v1:' + sha256(canonical_json(data).encode('utf-8')).hexdigest()
