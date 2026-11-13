"""Property identifiers bind scope/resource and receipts remain outputs only."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import ExecutionStatus, TruthStatus
from nonlocalgames.games import chsh
from nonlocalgames.properties import (PropertyAssessment, PropertyClaim, StructuralCounterexample,
                                      assess_property, property_assessment_to_dict,
                                      property_claim_from_dict, property_claim_to_dict)


@pytest.mark.parametrize('claim', [PropertyClaim('xor_predicate'),
    PropertyClaim('acceptance', off_support='sampled'),
    PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')),
    PropertyClaim('pseudotelepathy')])
def test_property_definition_roundtrip_and_name_independence(claim):
    game = chsh()
    data = json.loads(json.dumps(property_claim_to_dict(game, claim)))
    assert property_claim_from_dict(game, data) == claim
    assert property_claim_from_dict(replace(game, name='same verifier'), data) == claim


def test_changed_scope_kind_and_resource_invalidate_the_definition_identity():
    game = chsh()
    original = property_claim_to_dict(game, PropertyClaim('xor_predicate'))
    for change in (lambda p: p.update(off_support='sampled'), lambda p: p.update(kind='boolean_predicate'),
                   lambda p: p.update(unexpected=1)):
        data = deepcopy(original)
        change(data)
        with pytest.raises(ValueError):
            property_claim_from_dict(game, data)
    original = property_claim_to_dict(game, PropertyClaim('perfect_strategy', ResourceSpec('local')))
    original['resource']['kind'] = 'quantum_finite'
    with pytest.raises(ValueError):
        property_claim_from_dict(game, original)


def test_claims_and_assessment_snapshots_are_immutable():
    claim = PropertyClaim('xor_score')
    with pytest.raises(FrozenInstanceError):
        claim.kind = 'acceptance'
    profile = [0, 0]
    counter = StructuralCounterexample('test', profile)
    profile[0] = 1
    assert counter.input_profile == (0, 0)
    refs = ['first']
    receipt = PropertyAssessment('property', 'game', TruthStatus.UNKNOWN, ExecutionStatus.COMPLETED,
                                 verified_evidence=refs)
    refs.append('second')
    assert receipt.verified_evidence == ('first',)
    with pytest.raises(ValueError):
        PropertyAssessment('property', 'game', TruthStatus.PROVED, ExecutionStatus.INVALID_EVIDENCE)


def test_property_receipt_does_not_replace_raw_attainment_evidence():
    game = chsh()
    receipt = property_assessment_to_dict(assess_property(game, PropertyClaim('xor_predicate')))
    assert receipt['truth'] == 'proved'
    result = assess_property(game, PropertyClaim('perfect_strategy', ResourceSpec('quantum_finite')), [receipt])
    assert result.truth == TruthStatus.UNKNOWN and result.execution == ExecutionStatus.INVALID_EVIDENCE
