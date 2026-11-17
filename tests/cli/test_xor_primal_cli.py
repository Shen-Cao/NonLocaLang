import json

from nonlocalgames import AnalysisTask, ResourceSpec
from nonlocalgames.analysis import verify_xor_primal_certificate
from nonlocalgames.cli import main
from nonlocalgames.core.resource_serialization import dump_task
from nonlocalgames.games import chsh


def test_primal_gram_replays_through_both_cli_entry_points(tmp_path, capsys):
    game = chsh()
    task = AnalysisTask(game, ResourceSpec('quantum_finite'))
    assert 'xor_lower_certified' in task.capabilities()
    assert 'xor_lower_certified' not in AnalysisTask(game, ResourceSpec('quantum_finite', (2, 2))).capabilities()
    path = tmp_path / 'task.json'
    dump_task(task, path)
    for args in (['chsh', '--method', 'xor-lower-certified'],
                 ['task', '--file', str(path), '--method', 'xor_lower_certified']):
        main([*args, '--solver', 'SCS', '--witness'])
        record = json.loads(capsys.readouterr().out)['result']
        checked = verify_xor_primal_certificate(game, record['witness'])
        assert checked.value > 0.8 and record['value'] == str(checked.value)
