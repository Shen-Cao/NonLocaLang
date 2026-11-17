import json

import pytest

from nonlocalgames.claims import evidence_to_dict, make_evidence
from nonlocalgames.core.resources import ResourceSpec
from nonlocalgames.discovery import xor3_corpus as corpus
from nonlocalgames.discovery.xor3 import xor3_game
from nonlocalgames.discovery.xor3_evidence import representative_evidence


def _archive_prefix():
    """Build enough valid data to isolate errors in the first archived member."""
    representatives = {mask: representative_evidence(mask) for mask in (0, 1, 10)}
    return {
        **corpus._HEADER,
        'representatives': [{'mask': mask, 'evidence': [evidence_to_dict(a) for a in arts]}
                            for mask, arts in representatives.items()],
        'members': [corpus._checked_member(0, representatives)] + [{'mask': i} for i in range(1, 512)],
    }


@pytest.mark.parametrize('field', ['to_representative', 'orbit_id', 'verifier_id',
                                   'claim_id', 'target_evidence_ids', 'truth', 'local_value'])
def test_replay_rejects_tampered_member(tmp_path, field, monkeypatch):
    archive = _archive_prefix()
    archive['members'][0][field] = 'forged'
    path = tmp_path / 'corpus.json'
    path.write_text(json.dumps(archive), encoding='utf-8')
    monkeypatch.setattr(corpus, 'representative_evidence',
                        lambda *_: pytest.fail('replay must read the archived proof'))
    with pytest.raises(ValueError, match='differs.*mask 0'):
        corpus.replay_xor3_corpus(path)


def test_replay_rechecks_rehashed_raw_certificate(tmp_path):
    archive = _archive_prefix()
    payload = archive['representatives'][0]['evidence'][1]['payload']
    payload['upper_bound'] = '0'
    forged = make_evidence(xor3_game(0), ResourceSpec('quantum_commuting'), 'xor_dual', payload)
    archive['representatives'][0]['evidence'][1] = evidence_to_dict(forged)
    path = tmp_path / 'damaged-proof.json'
    path.write_text(json.dumps(archive), encoding='utf-8')
    with pytest.raises(ValueError, match='upper bound'):
        corpus.replay_xor3_corpus(path)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_type', 'wrong_query'])
def test_check_coverage_and_identity_before_transport(tmp_path, monkeypatch, mutation):
    archive = _archive_prefix()
    if mutation == 'missing':
        archive['members'].pop()
    elif mutation == 'duplicate':
        archive['members'][-1]['mask'] = 0
    elif mutation == 'wrong_type':
        archive['representatives'][0]['mask'] = False
    else:
        archive['query'] = {**archive['query'], 'threshold': '1/2'}
    path = tmp_path / 'wrong-family.json'
    path.write_text(json.dumps(archive), encoding='utf-8')
    monkeypatch.setattr(corpus, 'relabel_evidence', lambda *args: pytest.fail('check coverage first'))
    with pytest.raises(ValueError, match='coverage|ascending|query'):
        corpus.replay_xor3_corpus(path)


def test_bounded_archive_and_unique_json_fields(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{"outer":{"mask":1,"mask":2}}', encoding='utf-8')
    with pytest.raises(ValueError, match='max_file_bytes'):
        corpus.replay_xor3_corpus(path, max_file_bytes=5)
    with pytest.raises(ValueError, match='Duplicate'):
        corpus.replay_xor3_corpus(path)


def test_failed_write_preserves_prior_archive(tmp_path):
    from nonlocalgames.discovery._xor3_archive import write_archive
    path = tmp_path / 'corpus.json'
    write_archive(path, {'valid': 1})
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        write_archive(path, {'broken': float('nan')})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
