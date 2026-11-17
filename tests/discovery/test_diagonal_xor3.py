from fractions import Fraction as F

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import GapClaim, assess_gap, evidence_from_dict, evidence_to_dict
from nonlocalgames.discovery.diagonal_xor3 import diagonal_xor3_evidence, diagonal_xor3_game


@pytest.mark.parametrize('p', [F(0), F(1, 10), F(1999, 10000), F(1, 5), F(2001, 10000),
                               F(1, 4), F(1, 3), F(1, 2), F(3, 4), F(1)])
def test_strict_gap_and_both_optima_replay_from_raw_certificates(p):
    game = diagonal_xor3_game(p)
    quantum, local = ResourceSpec('quantum_finite'), ResourceSpec('local')
    artifacts = [evidence_from_dict(evidence_to_dict(a)) for a in diagonal_xor3_evidence(p)]
    result = assess_gap(game, GapClaim(quantum, local), artifacts)
    # Deterministic Bob signs are all equal or have one minority sign.
    classical_bias = max(abs(1-2*p), (1+2*p)/3)
    quantum_bias = max(abs(1-2*p), (1+p)/2)
    q_interval, l_interval = result.intervals
    assert result.execution == 'completed'
    assert l_interval.lower == l_interval.upper == (1+classical_bias)/2
    assert q_interval.lower == q_interval.upper == (1+quantum_bias)/2
    assert result.truth == ('proved' if F(1, 5) < p < 1 else 'disproved')


@pytest.mark.parametrize('p', [F(0), F(1, 8), F(1)])
def test_sampling_and_all_parity_rows_survive_at_endpoints(p):
    game = diagonal_xor3_game(p)
    for x in game.inputs:
        assert game.probability(x) == (p/3 if x[0] == x[1] else (1-p)/6)
        for a in game.outputs:
            assert game.reward(x, a) == int((sum(a) % 2) == int(x[0] == x[1]))


@pytest.mark.parametrize('p', [True, .5, F(-1, 10), '1.01', 'NaN'])
def test_family_requires_an_exact_probability(p):
    with pytest.raises((ValueError, TypeError)):
        diagonal_xor3_game(p)
