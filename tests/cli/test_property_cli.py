import json

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.analysis import classical_value, certify_quantum_strategy
from nonlocalgames.claims import EvidenceKind, evidence_to_dict, make_evidence
from nonlocalgames.cli import main
from nonlocalgames.games import ghz
from nonlocalgames.quantum.exact import exact_ghz_strategy


def test_structural_property_cli_is_exact_and_explicit(capsys):
    main(['property', 'ghz', '--kind', 'xor_predicate', '--off-support', 'sampled'])
    record = json.loads(capsys.readouterr().out)
    assert record['definition']['off_support'] == 'sampled'
    assert record['assessment']['truth'] == 'proved'
    assert record['assessment']['verified_evidence'] == []


def test_pseudotelepathy_cli_rechecks_raw_attainment_evidence(tmp_path, capsys):
    game = ghz()
    local = classical_value(game)
    quantum = certify_quantum_strategy(game, exact_ghz_strategy())
    artifacts = [make_evidence(game, ResourceSpec('local'), EvidenceKind.LOCAL_ENUMERATION,
                               {'value': str(local.value), 'strategy': local.witness}),
                 make_evidence(game, ResourceSpec('quantum_finite'), EvidenceKind.QUANTUM_STRATEGY,
                               quantum.witness)]
    path = tmp_path / 'evidence.json'
    path.write_text(json.dumps([evidence_to_dict(a) for a in artifacts]))
    main(['property', 'ghz', '--kind', 'pseudotelepathy', '--evidence', str(path)])
    record = json.loads(capsys.readouterr().out)
    assert record['assessment']['truth'] == 'proved'
    assert record['assessment']['execution'] == 'completed'
    assert len(record['assessment']['verified_evidence']) == 2


def test_property_cli_unknown_is_not_false_and_resources_are_explicit(capsys):
    main(['property', 'chsh', '--kind', 'perfect_strategy', '--resource', 'quantum_finite'])
    assert json.loads(capsys.readouterr().out)['assessment']['truth'] == 'unknown'
    for extra in ([], ['--resource', 'quantum_finite', '--off-support', 'all']):
        with pytest.raises(SystemExit):
            main(['property', 'chsh', '--kind', 'perfect_strategy', *extra])
        assert 'error:' in capsys.readouterr().err


def test_property_cli_rejects_receipts_and_wrong_context(tmp_path, capsys):
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps({'truth': 'proved', 'execution': 'completed'}))
    with pytest.raises(SystemExit):
        main(['property', 'ghz', '--kind', 'pseudotelepathy', '--evidence', str(path)])
    assert 'list of raw artifact' in capsys.readouterr().err
