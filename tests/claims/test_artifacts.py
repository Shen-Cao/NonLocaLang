"""Content identity and immutable evidence envelopes do not confer proof status."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from nonlocalgames import Game, ResourceSpec, Scenario
from nonlocalgames.claims.artifacts import (EvidenceArtifact, EvidenceKind, evidence_from_dict,
                                           evidence_to_dict, make_evidence)
from nonlocalgames.core.identity import canonical_json, resource_id, verifier_digest, verifier_id
from nonlocalgames.core.resource_serialization import resource_to_dict
from nonlocalgames.games import chsh
from nonlocalgames.quantum import FixedState


def test_verifier_identity_preserves_legacy_digest_and_ignores_display_name():
    game = chsh()
    legacy = {'input_sizes': game.scenario.input_sizes, 'output_sizes': game.scenario.output_sizes,
              'distribution': tuple(map(str, game.distribution)), 'payoff': tuple(map(str, game.payoff))}
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert verifier_digest(game) == digest
    assert verifier_id(game) == 'nonlocalgames.verifier.v1:' + digest
    assert verifier_id(replace(game, name='same verifier, different name')) == verifier_id(game)
    resource = ResourceSpec('local')
    first = make_evidence(game, resource, 'local_strategy', {'strategy': [[0, 0], [0, 0]]})
    second = make_evidence(replace(game, name='renamed'), resource, 'local_strategy', first.payload)
    assert first.artifact_id == second.artifact_id


def test_equal_score_coefficients_do_not_identify_different_sampling_verifiers():
    scenario = Scenario((2,), (1,))
    first = Game(scenario, (Fraction(1, 2), Fraction(1, 2)), (2, 2))
    second = Game(scenario, (Fraction(1, 4), Fraction(3, 4)), (4, Fraction(4, 3)))
    assert first.coefficients == second.coefficients
    assert verifier_id(first) != verifier_id(second)


def test_resource_identity_binds_restrictions_without_approximate_equivalence():
    first = ResourceSpec('quantum_finite', (2, 2))
    second = ResourceSpec('quantum_finite', (3, 3))
    assert resource_id(first) != resource_id(second)
    assert resource_id(ResourceSpec('local')) != resource_id(ResourceSpec('local', shared_randomness='forbidden'))
    pure = ResourceSpec('quantum_finite', fixed_state=FixedState(np.diag([1, 0]), (2,)))
    nearby = ResourceSpec('quantum_finite', fixed_state=FixedState(np.diag([1-1e-12, 1e-12]), (2,)))
    assert resource_id(pure) != resource_id(nearby)
    expected = hashlib.sha256(canonical_json(resource_to_dict(first)).encode()).hexdigest()
    assert resource_id(first) == 'nonlocalgames.resource.v2:' + expected


def test_payload_and_resource_snapshots_are_immutable_and_decode_freshly():
    payload = {'strategy': [[0, 0], [0, 0]]}
    artifact = make_evidence(chsh(), ResourceSpec('local'), EvidenceKind.LOCAL_STRATEGY, payload)
    identity = artifact.artifact_id
    payload['strategy'][0][0] = 1
    artifact.payload['strategy'][1][0] = 1
    assert artifact.payload['strategy'] == [[0, 0], [0, 0]]
    assert artifact.artifact_id == identity
    assert artifact.resource is not artifact.resource
    with pytest.raises(FrozenInstanceError):
        artifact.payload_json = '{}'
    restored = evidence_from_dict(json.loads(json.dumps(evidence_to_dict(artifact))))
    assert restored == artifact
    assert restored.nbytes == len(canonical_json(evidence_to_dict(restored)).encode())


def test_content_id_access_does_not_revalidate_quantum_state(monkeypatch):
    resource = ResourceSpec('quantum_finite', fixed_state=FixedState(np.eye(4)/4, (2, 2)))
    artifact = make_evidence(chsh(), resource, 'numerical_hint', {'score': .5})
    expected_resource, expected_artifact = artifact.resource_id, artifact.artifact_id
    monkeypatch.setattr(np.linalg, 'eigvalsh', lambda *a, **k: pytest.fail('IDs must not decode a quantum state'))
    assert artifact.resource_id == expected_resource
    assert artifact.artifact_id == expected_artifact
    assert evidence_to_dict(artifact)['resource_id'] == expected_resource


def test_valid_envelopes_and_claimed_exact_metadata_are_still_untrusted():
    artifact = make_evidence(chsh(), ResourceSpec('quantum_finite'), 'numerical_hint',
                             {'guarantee': 'exact', 'value': 1000, 'certified': True})
    restored = evidence_from_dict(evidence_to_dict(artifact))
    assert restored.payload['guarantee'] == 'exact'
    assert restored.kind is EvidenceKind.NUMERICAL_HINT
    assert 'not authenticated' in EvidenceArtifact.__doc__


def test_envelope_rejects_tampered_fields_and_content_ids():
    artifact = make_evidence(chsh(), ResourceSpec('local'), 'local_strategy', {'strategy': [[0, 0], [0, 0]]})
    envelope = evidence_to_dict(artifact)
    edits = {'game_id': 'nonlocalgames.verifier.v1:' + '0'*64,
             'resource_id': 'nonlocalgames.resource.v2:' + '0'*64,
             'artifact_id': 'nonlocalgames.evidence.v1:' + '0'*64,
             'kind': 'xor_dual', 'payload': {'strategy': [[1, 0], [0, 0]]}, 'extra': True}
    for key, value in edits.items():
        with pytest.raises(ValueError):
            evidence_from_dict({**envelope, key: value})
    for key in envelope:
        incomplete = deepcopy(envelope)
        incomplete.pop(key)
        with pytest.raises(ValueError):
            evidence_from_dict(incomplete)
    with pytest.raises(ValueError, match='identifier'):
        EvidenceArtifact('local_strategy', 'wrong', artifact.resource_json, artifact.payload_json)
    with pytest.raises(ValueError):
        make_evidence(chsh(), ResourceSpec('local'), 'unsupported_evidence_kind', {})


def test_json_payload_types_are_explicit_and_no_rationalization_occurs():
    assert canonical_json({'z': ('1/2', '1/2'), 'a': 1}) == '{"a":1,"z":["1/2","1/2"]}'
    for payload in ({'value': Fraction(1, 2)}, {'value': {1, 2}}, {1: 'nonstring key'},
                    {'vector': (v for v in range(2))}):
        with pytest.raises(TypeError):
            make_evidence(chsh(), ResourceSpec('local'), 'numerical_hint', payload)
    for value in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            make_evidence(chsh(), ResourceSpec('local'), 'numerical_hint', {'value': value})
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match='cyclic'):
        canonical_json(cyclic)
    artifact = make_evidence(chsh(), ResourceSpec('local'), 'numerical_hint', {})
    with pytest.raises(ValueError, match='Duplicate'):
        EvidenceArtifact('numerical_hint', artifact.game_id, artifact.resource_json, '{"a":1,"a":2}')


def test_full_encoded_artifact_budget_is_checked_without_solver_work():
    artifact = make_evidence(chsh(), ResourceSpec('local'), 'numerical_hint', {'note': 'small'})
    envelope = evidence_to_dict(artifact)
    assert evidence_from_dict(envelope, max_bytes=artifact.nbytes) == artifact
    for operation in (
        lambda: evidence_from_dict(envelope, max_bytes=artifact.nbytes-1),
        lambda: evidence_to_dict(artifact, max_bytes=artifact.nbytes-1),
        lambda: make_evidence(chsh(), ResourceSpec('local'), 'numerical_hint', {'note': 'x'*1000}, max_bytes=100),
    ):
        with pytest.raises(ValueError, match='max_bytes'):
            operation()
    for budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            evidence_from_dict(envelope, max_bytes=budget)


def test_noncanonical_legacy_resource_cannot_silently_enter_an_artifact():
    artifact = make_evidence(chsh(), ResourceSpec('local'), 'numerical_hint', {})
    legacy = resource_to_dict(ResourceSpec('local'))
    legacy['schema'] = 'nonlocalgames.resource.v1'
    legacy.pop('network')
    with pytest.raises(ValueError, match='current resource codec'):
        EvidenceArtifact('numerical_hint', artifact.game_id, canonical_json(legacy), '{}')


def test_signed_zero_resource_components_roundtrip_without_identity_drift():
    from nonlocalgames.quantum import chsh_strategy
    state = chsh_strategy().state.conjugate()
    assert np.signbit(state.imag).any()
    resource = ResourceSpec('quantum_finite', fixed_state=FixedState(state, (2, 2)))
    artifact = make_evidence(chsh(), resource, 'numerical_hint', {})
    assert artifact.resource_id == resource_id(resource) == resource_id(artifact.resource)
    decoded = evidence_from_dict(evidence_to_dict(artifact))
    assert decoded.resource_json == artifact.resource_json
    assert decoded.artifact_id == artifact.artifact_id
    np.testing.assert_array_equal(np.signbit(decoded.resource.fixed_state.state.imag), np.signbit(state.imag))


def test_game_id_is_a_content_binding_not_an_authenticated_game_lookup():
    game = chsh()
    different = replace(game, payoff=tuple(1-value for value in game.payoff))
    first = make_evidence(game, ResourceSpec('local'), 'numerical_hint', {'value': .5})
    second = make_evidence(different, ResourceSpec('local'), 'numerical_hint', first.payload)
    assert first.game_id != second.game_id and first.artifact_id != second.artifact_id
    unresolved = EvidenceArtifact(first.kind, 'nonlocalgames.verifier.v1:' + '0'*64,
                                   first.resource_json, first.payload_json)
    assert evidence_from_dict(evidence_to_dict(unresolved)).game_id == unresolved.game_id
    assert unresolved.game_id != verifier_id(game)
