import copy
import json
from fractions import Fraction

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import (GapClaim, ValueClaim, assessment_to_dict, assess_gap,
                                  claim_from_dict, claim_to_dict)
from nonlocalgames.games import chsh, biased_chsh


@pytest.mark.parametrize('claim', [ValueClaim(ResourceSpec('quantum_finite'), 'at_most', Fraction(9, 10)),
                                   GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local'))])
def test_claim_definition_roundtrip_and_context_binding(claim):
    game = chsh()
    data = json.loads(json.dumps(claim_to_dict(game, claim)))
    assert claim_from_dict(game, data) == claim
    with pytest.raises(ValueError, match='different verifier'):
        claim_from_dict(biased_chsh(Fraction(1, 3)), data)
    changed = copy.deepcopy(data)
    changed['threshold'] = '1/2'
    with pytest.raises(ValueError, match='identity'):
        claim_from_dict(game, changed)
    changed = {**data, 'threshold': .1}
    with pytest.raises(ValueError, match='rational strings'):
        claim_from_dict(game, changed)


def test_assessment_is_portable_receipt_not_a_new_evidence_source():
    game = chsh()
    claim = GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local'))
    result = assess_gap(game, claim)
    data = assessment_to_dict(result)
    assert json.loads(json.dumps(data))['truth'] == 'unknown'
    refused = assess_gap(game, claim, [result])
    assert refused.execution == 'invalid_evidence' and refused.truth == 'unknown'
