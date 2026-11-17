"""Public claim replay for each calibration representative's portable evidence."""
from fractions import Fraction

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import GapClaim, assess_gap, evidence_from_dict, evidence_to_dict
from nonlocalgames.discovery.xor3 import xor3_game
from nonlocalgames.discovery.xor3_evidence import representative_evidence


@pytest.mark.parametrize('mask,tight,local,lower,upper,gap', [
    (0, False, 1, 1, 1, 'disproved'),
    (1, False, Fraction(8, 9), Fraction(401, 450), None, 'proved'),
    (10, False, Fraction(7, 9), Fraction(187, 225), None, 'proved'),
    (10, True, Fraction(7, 9), Fraction(5, 6), Fraction(5, 6), 'proved'),
])
def test_representative_evidence_replays(mask, tight, local, lower, upper, gap):
    claim = GapClaim(ResourceSpec('quantum_finite'), ResourceSpec('local'))
    artifacts = representative_evidence(mask, tight=tight)
    restored = [evidence_from_dict(evidence_to_dict(artifact)) for artifact in artifacts]
    result = assess_gap(xor3_game(mask), claim, restored)
    assert (result.execution, result.truth) == ('completed', gap)
    quantum_interval, local_interval = result.intervals
    assert local_interval.lower == local_interval.upper == local
    assert (quantum_interval.lower, quantum_interval.upper) == (lower, upper)


def test_optimum_and_representative_scope_are_explicit():
    with pytest.raises(NotImplementedError, match='optimum'):
        representative_evidence(1, tight=True)
    with pytest.raises(ValueError, match='representatives'):
        representative_evidence(2)
