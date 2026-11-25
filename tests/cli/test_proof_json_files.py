import json

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import ValueClaim, claim_to_dict, evidence_to_dict, make_evidence
from nonlocalgames.cli import main
from nonlocalgames.core.resource_serialization import resource_to_dict
from nonlocalgames.core.serialization import to_dict
from nonlocalgames.games import chsh


@pytest.fixture
def files(tmp_path):
    game, resource = chsh(), ResourceSpec('local')
    artifact = make_evidence(game, resource, 'local_strategy', {'strategy': [[0, 0], [0, 0]]})
    docs = {'game': to_dict(game), 'resource': resource_to_dict(resource), 'budget': {},
            'definition': claim_to_dict(game, ValueClaim(resource, 'at_least', '3/4')),
            'evidence': [evidence_to_dict(artifact)]}
    paths = {}
    for role, data in docs.items():
        path = paths[role] = tmp_path / (role + '.json')
        path.write_text(json.dumps(data), encoding='utf-8')
    return paths


def run_cli(command, files):
    if command == 'claim':
        main(['claim', str(files['game']), str(files['definition']), str(files['evidence']),
              '--budget', str(files['budget'])])
    else:
        main(['property', 'json', '--file', str(files['game']), '--kind', 'perfect_strategy',
              '--resource-file', str(files['resource']), '--evidence', str(files['evidence'])])


@pytest.mark.parametrize(('command', 'role'), [
    ('claim', 'game'), ('claim', 'definition'), ('claim', 'evidence'), ('claim', 'budget'),
    ('property', 'game'), ('property', 'resource'), ('property', 'evidence')])
def test_all_external_claim_and_property_inputs_reject_duplicate_names(command, role, files, capsys):
    # Escapes do not make a different member name. Exercise objects inside arrays as well.
    duplicate = r'{"key":0,"\u006bey":0}'
    files[role].write_text('[{"payload":' + duplicate + '}]' if role == 'evidence' else duplicate,
                           encoding='utf-8')
    with pytest.raises(SystemExit) as error:
        run_cli(command, files)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "Duplicate JSON key 'key'" in captured.err and not captured.out


def test_valid_claim_and_property_files_replay_with_the_original_content_ids(files, capsys):
    run_cli('claim', files)
    claim = json.loads(capsys.readouterr().out)
    # Legacy property inputs can also be UTF-16/32 or have a UTF-8 BOM.
    for role, encoding in (('game', 'utf-16'), ('resource', 'utf-32-be'), ('evidence', 'utf-8-sig')):
        files[role].write_bytes(files[role].read_text(encoding='utf-8').encode(encoding))
    run_cli('property', files)
    prop = json.loads(capsys.readouterr().out)['assessment']

    # The deterministic strategy attains 3/4, not a perfect score.
    assert claim['truth'] == 'proved' and prop['truth'] == 'unknown'
    assert claim['checks'][0]['attained_score'] == prop['checks'][0]['attained_score'] == '3/4'
    assert claim['game_id'] == prop['game_id'] == (
        'nonlocalgames.verifier.v1:0307edfe8e0a5b1798575be20066f5c2316cd83a1a874acb41bea17937afc043')
    assert claim['claim_id'] == (
        'nonlocalgames.claim.v1:01bc3f4482b41ecd1f7a13dc19afc8781c0490973dcabd3395931a783460fc6e')
    assert prop['property_id'] == (
        'nonlocalgames.property.v1:2e25712f315feb35c780a6c4a36aa49afad5aa9825cb16cd90589198b1fdf45a')
    assert claim['verified_evidence'] == prop['verified_evidence'] == [
        'nonlocalgames.evidence.v1:838b8b6bcd8f1099d04e723e1ef1945e3d870dea86c68618fb0c28fbfb555171']
