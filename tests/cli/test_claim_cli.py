import json

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import ValueClaim, claim_to_dict, evidence_to_dict
from nonlocalgames.cli import main
from nonlocalgames.core.serialization import to_dict
from nonlocalgames.discovery import representative_evidence, xor3_game


def test_check_portable_value_claim_with_dimension_bound(tmp_path, capsys):
    game = xor3_game(1)
    claim = ValueClaim(ResourceSpec('quantum_finite', (2, 2)), 'at_least', '401/450')
    args = ['claim']
    for name, data in (('game', to_dict(game)), ('claim', claim_to_dict(game, claim)),
                       ('evidence', [evidence_to_dict(a) for a in representative_evidence(1)])):
        path = tmp_path / (name+'.json')
        path.write_text(json.dumps(data))
        args.append(str(path))
    main(args)
    record = json.loads(capsys.readouterr().out)
    assert record['truth'] == 'proved' and record['local_dimensions'] == [2, 2]
    assert record['intervals'][0]['lower'] == '401/450'
