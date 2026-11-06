import json
from fractions import Fraction

import pytest

from nonlocalgames import NetworkCompatibilityTask, ResourceSpec, Scenario
from nonlocalgames.cli import main
from nonlocalgames.core.compatibility_serialization import dump_compatibility
from nonlocalgames.networks import RationalBehavior, SourceNetwork
from nonlocalgames.networks.inflation import verify_spiral_certificate
from nonlocalgames.networks.serialization import classical_from_dict


def make_task(path, *, kind='local', chain=False):
    s = Scenario((1, 1, 1), (2, 2, 2))
    p = (Fraction(1, 8),)*8 if chain else tuple(Fraction(1, 3) if sum(a) == 1 else 0 for a in s.outputs)
    graph = SourceNetwork(3, ((0, 1), (1, 2)) if chain else ((0, 1), (1, 2), (2, 0)))
    task = NetworkCompatibilityTask(RationalBehavior(s, p), ResourceSpec(kind, network=graph,
                                   shared_randomness='forbidden'))
    dump_compatibility(task, path)
    return task


def test_cli_spiral_certificate_roundtrip(tmp_path, capsys):
    path = tmp_path / 'w.json'
    task = make_task(path)
    main(['network', '--file', str(path), '--method', 'spiral', '--witness'])
    record = json.loads(capsys.readouterr().out)
    result = record['result']
    assert result['resource'] == 'classical_fanout'
    checked = verify_spiral_certificate(task.resource.network, task.behavior, result['certificate'])
    assert checked.status == 'incompatible'
    assert checked.guarantee == result['guarantee'] == 'exact_classical_incompatibility'


def test_cli_preserves_quantum_scope_and_does_not_silently_choose_method(tmp_path, capsys):
    path = tmp_path / 'q.json'
    make_task(path, kind='quantum_finite')
    main(['network', '--file', str(path), '--capabilities'])
    assert 'spiral' not in json.loads(capsys.readouterr().out)['capabilities']
    for rest in ([], ['--method', 'spiral'], ['--method', 'triangle_finner', '--time-limit', '1']):
        with pytest.raises(SystemExit) as error:
            main(['network', '--file', str(path), *rest])
        assert error.value.code == 2
        assert 'error:' in capsys.readouterr().err


def test_cli_chain_constructive_witness_roundtrip(tmp_path, capsys):
    path = tmp_path / 'chain.json'
    task = make_task(path, chain=True)
    main(['network', '--file', str(path), '--method', 'bilocal_chain', '--witness'])
    result = json.loads(capsys.readouterr().out)['result']
    assert result['status'] == 'compatible'
    assert classical_from_dict(result['witness']).probabilities() == task.behavior.probabilities
