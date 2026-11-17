from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from threading import Barrier
from types import SimpleNamespace

import pytest

from nonlocalgames import ResourceSpec
from nonlocalgames.claims import VerificationBudget, evidence_to_dict, make_evidence
from nonlocalgames.claims.types import EvidenceError
from nonlocalgames.core.identity import canonical_json
from nonlocalgames.discovery import _weighted_record as records
from nonlocalgames.discovery import weighted_run as runner
from nonlocalgames.games import biased_chsh


def _run(path, **overrides):
    return runner.run_weighted_chsh(path, **{'steps': 4, 'screen': False, **overrides})


def test_partial_run_resumes_with_saved_evidence_and_replays_without_discovery(tmp_path, monkeypatch):
    partial = _run(tmp_path, max_new=2)
    assert partial['pending'] == [2, 3, 4] and not partial['coverage_complete']
    assert runner.replay_weighted_chsh(tmp_path) == partial
    first = (tmp_path / 'members/0.json').read_bytes()
    original = runner.generate_member
    visited = []
    def generate(manifest, k, budget):
        visited.append(k)
        return original(manifest, k, budget)
    monkeypatch.setattr(runner, 'generate_member', generate)
    final = _run(tmp_path)
    assert final['coverage_complete'] and final['pending'] == [] and visited == [2, 3, 4]
    assert (tmp_path / 'members/0.json').read_bytes() == first
    monkeypatch.setattr(records, 'weighted_chsh_member', lambda *a, **kw: pytest.fail('no discovery on replay'))
    monkeypatch.setattr(records, 'xor_quantum_value', lambda *a, **kw: pytest.fail('no solver on replay'))
    review = runner.replay_weighted_chsh(tmp_path, include_members=True)
    rows = review.pop('results')
    assert review == final
    assert rows[0]['truth'] == 'disproved' and rows[0]['certified_gap_lower'] == '0'
    assert rows[1]['p'] == '1/4' and rows[1]['truth'] == 'proved'
    assert _run(tmp_path) == final


def test_interruption_before_a_publish_leaves_complete_prefix(tmp_path, monkeypatch):
    real = runner.generate_member
    def interrupted(manifest, k, budget):
        if k == 1:
            raise KeyboardInterrupt
        return real(manifest, k, budget)
    monkeypatch.setattr(runner, 'generate_member', interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path)
    (tmp_path / 'members/.nlg-checkpoint-example.tmp').write_text('incomplete')
    assert runner.replay_weighted_chsh(tmp_path)['pending'] == [1, 2, 3, 4]
    monkeypatch.setattr(runner, 'generate_member', real)
    assert _run(tmp_path)['recorded'] == 5


def test_competing_writers_keep_exactly_one_valid_record_per_parameter(tmp_path, monkeypatch):
    original = runner.generate_member
    both_here = Barrier(2)
    def overlap(manifest, k, budget):
        if k == 0:
            both_here.wait(timeout=10)
        return original(manifest, k, budget)
    monkeypatch.setattr(runner, 'generate_member', overlap)
    with ThreadPoolExecutor(max_workers=2) as workers:
        saved = list(workers.map(lambda _: _run(tmp_path, steps=1), range(2)))
    assert saved[0] == saved[1] == runner.replay_weighted_chsh(tmp_path)
    assert sorted(p.name for p in (tmp_path / 'members').iterdir()) == ['0.json', '1.json']


def test_run_settings_mismatch_preserves_completed_members(tmp_path):
    _run(tmp_path, max_new=1)
    before = (tmp_path / 'members/0.json').read_bytes()
    with pytest.raises(ValueError, match='manifest differs'):
        _run(tmp_path, grid_denominator=1)
    assert (tmp_path / 'members/0.json').read_bytes() == before


def test_generation_exhaustion_does_not_assert_a_bound(tmp_path):
    result = _run(tmp_path, steps=1, verification_budget=VerificationBudget(max_strategies=1))
    assert result['generation_failures'] == result['verdicts']['unknown'] == 2
    row = runner.replay_weighted_chsh(tmp_path, include_members=True)['results'][0]
    assert row['execution'] == 'budget_exceeded' and row['truth'] == 'unknown'
    assert row['local_value'] is None and row['quantum_lower'] is None
    assert row['generation_error']['status'] == 'budget_exceeded'


@pytest.mark.parametrize('budget', [VerificationBudget(max_strategies=1),
                                    VerificationBudget(max_artifact_bytes=32)])
def test_tighter_replay_limits_report_exhaustion(tmp_path, budget):
    _run(tmp_path, max_new=1)
    with pytest.raises(EvidenceError) as error:
        runner.replay_weighted_chsh(tmp_path, budget=budget)
    assert error.value.issue.code == 'budget_exceeded'


def test_solver_unavailable_preserves_exact_results(tmp_path):
    result = _run(tmp_path, max_new=1, screen=True, solver='MISSING_SOLVER')
    assert result['screen_statuses'] == {'unavailable': 1}
    assert result['verdicts']['disproved'] == 1
    assert runner.replay_weighted_chsh(tmp_path) == result


def test_positive_numerical_gap_and_solver_failure_leave_independent_verdicts(tmp_path, monkeypatch):
    import cvxpy
    monkeypatch.setattr(cvxpy, 'installed_solvers', lambda: ['SCS'])
    def screen(game, *, solver, eps):
        assert solver == 'SCS' and eps == 1e-8
        if game.distribution[-1] == 0:
            raise RuntimeError('synthetic solver failure')
        return SimpleNamespace(value=0.90000000001, status='optimal', diagnostics={'solver': solver})
    monkeypatch.setattr(records, 'xor_quantum_value', screen)
    result = _run(tmp_path, steps=10, max_new=2, screen=True)
    assert result['screen_statuses'] == {'completed': 1, 'failed': 1}
    monkeypatch.setattr(records, 'xor_quantum_value', lambda *a, **kw: pytest.fail('solver called on replay'))
    rows = runner.replay_weighted_chsh(tmp_path, include_members=True)['results']
    assert rows[0]['truth'] == 'disproved'
    assert rows[1]['p'] == '1/10' and rows[1]['candidate_gap'] > 0
    assert rows[1]['truth'] == 'disproved' and rows[1]['certified_gap_lower'] == '0'


def test_coarse_grid_retains_unresolved_case(tmp_path):
    _run(tmp_path, max_new=2, grid_denominator=1)
    row = runner.replay_weighted_chsh(tmp_path, include_members=True)['results'][1]
    assert row['p'] == '1/4' and row['truth'] == 'unknown'
    assert row['execution'] == 'completed' and row['generation_error'] is None


@pytest.mark.parametrize('mutation', ['forged_verdict', 'rehashed_bad_proof', 'p', 'run_id', 'screen', 'file_id'])
def test_replay_rejects_inconsistent_or_forged_records(tmp_path, mutation):
    _run(tmp_path, max_new=1)
    path = tmp_path / 'members/0.json'
    record = json.loads(path.read_bytes())
    if mutation == 'forged_verdict':
        record['assessment']['truth'] = 'proved'
    elif mutation == 'rehashed_bad_proof':
        bad = record['evidence'][0]['payload']
        bad['value'] = '0'
        record['evidence'][0] = evidence_to_dict(make_evidence(
            biased_chsh(0), ResourceSpec('local'), 'local_enumeration', bad))
    elif mutation == 'p':
        record['p'] = '1/2'
    elif mutation == 'run_id':
        record['run_id'] = 'wrong'
    elif mutation == 'screen':
        record['screen']['value'] = 1
    if mutation == 'file_id':
        record['member_id'] = 'wrong'
    else:
        del record['member_id']
        record['member_id'] = record['schema'] + ':' + sha256(canonical_json(record).encode()).hexdigest()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        runner.replay_weighted_chsh(tmp_path)


@pytest.mark.parametrize('mutation', ['hole', 'extra', 'duplicate_keys', 'too_large', 'missing_manifest'])
def test_checkpoint_coverage_and_shape_errors_are_rejected(tmp_path, mutation, monkeypatch):
    _run(tmp_path, max_new=2)
    path = tmp_path / 'members/0.json'
    if mutation == 'hole':
        path.unlink()
    elif mutation == 'extra':
        (tmp_path / 'members/01.json').write_text('{}')
    elif mutation == 'duplicate_keys':
        path.write_text('{"k":0,"k":0}')
    elif mutation == 'too_large':
        monkeypatch.setattr(runner, 'MEMBER_BYTES', 32)
    else:
        (tmp_path / 'manifest.json').unlink()
        with pytest.raises(ValueError, match='without their manifest'):
            _run(tmp_path)
        return
    with pytest.raises(ValueError):
        runner.replay_weighted_chsh(tmp_path)
