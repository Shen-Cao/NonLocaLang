from fractions import Fraction as F

import pytest

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.certificates.xor_primal import verify_xor_gram, verify_xor_primal_certificate
from nonlocalgames.claims import make_evidence, verify_artifact
from nonlocalgames.claims.types import EvidenceError
from nonlocalgames.equivalence import Relabelling, relabel_evidence, relabel_game


def _gram(vectors):
    return tuple(tuple(sum(a*b for a, b in zip(v, w)) for w in vectors) for v in vectors)


def _source():
    scenario = Scenario((2, 3), (2, 2))
    distribution = dict(zip(scenario.inputs, (F(w, 28) for w in (0, 2, 3, 5, 7, 11))))
    def score(x, a):
        if x == (0, 0):
            return a[0]
        return 2 if (a[0] ^ a[1]) == (x[0]+x[1]) % 2 else -1
    game = Game.from_rule(scenario, score, distribution=distribution)
    vectors = [(0, 1), (F(4, 5), F(3, 5)), (F(-12, 13), F(5, 13)), (0, -1), (F(3, 5), F(-4, 5))]
    point = verify_xor_gram(game, _gram(vectors))
    artifact = make_evidence(game, ResourceSpec('quantum_finite'), 'xor_primal', point.witness)
    mapping = Relabelling(scenario, (1, 0), ((2, 0, 1), (1, 0)),
                          (((1, 0), (0, 1), (1, 0)), ((0, 1), (1, 0))))
    return game, mapping, point, artifact


def test_primal_transport_signed_node_permutation_preserves_rank_and_attained_score():
    game, mapping, point, artifact = _source()
    target = relabel_game(game, mapping)
    moved = relabel_evidence(game, artifact, mapping).artifact
    assert moved.game_id != artifact.game_id and moved.resource_id == artifact.resource_id
    # Bob's signed nodes (-B2, B0, -B1) precede Alice's nodes (A1, -A0).
    target_vectors = [(F(-3, 5), F(4, 5)), (F(-12, 13), F(5, 13)), (0, 1),
                      (F(4, 5), F(3, 5)), (0, -1)]
    assert moved.payload['gram'] == [list(map(str, row)) for row in _gram(target_vectors)]
    checked = verify_xor_primal_certificate(target, moved.payload)
    assert checked.value == point.value
    assert checked.diagnostics['gram_rank'] == point.diagnostics['gram_rank'] == 2
    assert checked.diagnostics['realization_dimensions'] == point.diagnostics['realization_dimensions'] == (2, 2)
    facts = verify_artifact(target, moved)
    assert facts.lower == facts.attained_score == point.value
    assert facts.actual_dimensions == (2, 2) and facts.upper is None
    assert relabel_evidence(target, moved, mapping.inverse()).artifact == artifact


def test_invalid_primal_source_data_or_resource_are_rejected_before_transport(monkeypatch):
    import nonlocalgames.equivalence.evidence as transport
    game, mapping, _, artifact = _source()
    monkeypatch.setattr(transport, '_xor_payload', lambda *args: pytest.fail('source must pass first'))
    tampered = make_evidence(game, ResourceSpec('quantum_finite'), 'xor_primal',
                             {**artifact.payload, 'achieved_score': str(F(artifact.payload['achieved_score'])-1)})
    wrong_resource = make_evidence(game, ResourceSpec('nonsignalling'), 'xor_primal', artifact.payload)
    for source, expected in ((tampered, 'invalid_evidence'), (wrong_resource, 'unsupported')):
        with pytest.raises(EvidenceError) as caught:
            relabel_evidence(game, source, mapping)
        assert caught.value.issue.code == expected
