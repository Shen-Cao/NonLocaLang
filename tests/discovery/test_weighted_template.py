from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys

import pytest

from nonlocalgames.analysis import xor_template
from nonlocalgames.core.identity import canonical_json
from nonlocalgames.discovery import _weighted_record as records
from nonlocalgames.discovery import run_manifest as manifests
from nonlocalgames.discovery.weighted_run import replay_weighted_chsh, run_weighted_chsh


def test_default_manifest_retains_archived_v1_encoding(monkeypatch):
    provenance = {'source_schema': 'nonlocalgames.python_source_tree.v1', 'source_sha256': 'a'*64,
                  'git_revision': 'b'*40, 'python': '3.12', 'implementation': 'CPython',
                  'packages': {k: 'fixture' for k in ('numpy', 'scipy', 'cvxpy', 'nonlocalgames', 'sympy')},
                  'solver_distribution': 'scs', 'solver_version': '3'}
    monkeypatch.setattr(manifests, 'runtime_provenance', lambda _: provenance)
    old = manifests.create_manifest(steps=2)
    assert old['run_id'] == (manifests.RUN_SCHEMA + ':'
                             '0aaf8a572c627d116d900067df49184ec790add7e30740f1cb0a8411f1c0c0f8')
    monkeypatch.setattr(manifests, 'runtime_provenance', lambda _: pytest.fail('read saved provenance'))
    assert manifests.validate_manifest(json.loads(json.dumps(old))) == old


def test_template_manifest_binds_fresh_solver_state_and_effective_options():
    manifest = manifests.create_manifest(steps=2, reuse_screen=True)
    assert manifest['schema'] == manifests.TEMPLATE_RUN_SCHEMA
    assert manifest['template_policy'] == 'fresh_solver_state'
    assert manifest['solver_options'] == {'eps': 1e-8, 'warm_start': False}
    assert manifests.validate_manifest(json.loads(json.dumps(manifest))) == manifest
    for field, value in [('solver_options', {'eps': 1e-8, 'warm_start': True}),
                         ('template_policy', 'warm'), ('screen_method', 'xor_vector_sdp.v1')]:
        changed = deepcopy(manifest)
        del changed['run_id']
        changed[field] = value
        changed['run_id'] = changed['schema'] + ':' + sha256(canonical_json(changed).encode()).hexdigest()
        with pytest.raises(ValueError, match='settings or run identity'):
            manifests.validate_manifest(changed)
    with pytest.raises(ValueError, match='requires enabled'):
        manifests.create_manifest(reuse_screen=True, screen=False)
    with pytest.raises(ValueError, match='Boolean'):
        manifests.create_manifest(reuse_screen=1)


def test_one_session_per_invocation_with_pending_work_and_solver_free_replay(tmp_path, monkeypatch):
    instances = []
    real = xor_template.XorScreenSession

    def new_session(scenario, **options):
        instances.append(options)
        return real(scenario, **options)

    monkeypatch.setattr(xor_template, 'XorScreenSession', new_session)
    monkeypatch.setattr(records, 'xor_quantum_value', lambda *a, **k: pytest.fail('unexpected ordinary solve'))
    options = dict(steps=4, reuse_screen=True)
    assert run_weighted_chsh(tmp_path, **options, max_new=0)['recorded'] == 0
    assert not instances
    first = run_weighted_chsh(tmp_path, **options, max_new=3)
    assert first['screen_statuses'] == {'completed': 3}
    assert len(instances) == 1
    complete = run_weighted_chsh(tmp_path, **options)
    assert complete['screen_statuses'] == {'completed': 5}
    assert len(instances) == 2
    assert all(o == {'eps': 1e-8, 'warm_start': False, 'solver': 'SCS'} for o in instances)
    monkeypatch.setattr(xor_template, 'XorScreenSession', lambda *a, **k: pytest.fail('unexpected session'))
    assert run_weighted_chsh(tmp_path, **options) == complete
    with pytest.raises(ValueError, match='manifest differs'):
        run_weighted_chsh(tmp_path, steps=4)
    script = """
import json
import sys
import cvxpy
def forbidden(*args, **kwargs):
    raise AssertionError('Replay initialized a numerical problem')
cvxpy.Problem.__init__ = forbidden
from nonlocalgames.discovery.weighted_run import replay_weighted_chsh
print(json.dumps(replay_weighted_chsh(sys.argv[1])))
"""
    process = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True,
                             check=True, text=True)
    assert json.loads(process.stdout) == complete


@pytest.mark.parametrize('policy, expected', [('missing', 'unavailable'), ('init_error', 'failed'),
                                             ('solve_error', 'failed')])
def test_template_failures_preserve_checked_evidence_and_never_fallback(tmp_path, monkeypatch, policy, expected):
    calls = []
    class FailedSession:
        def __init__(self, *args, **kwargs):
            calls.append('build')
            if policy == 'init_error':
                raise RuntimeError('synthetic initialization failure')

        def solve(self, game):
            calls.append('solve')
            raise RuntimeError('synthetic solve failure')
    monkeypatch.setattr(xor_template, 'XorScreenSession', FailedSession)
    monkeypatch.setattr(records, 'xor_quantum_value', lambda *a, **k: pytest.fail('unexpected fallback'))
    solver = 'ABSENT' if policy == 'missing' else 'SCS'
    generated = run_weighted_chsh(tmp_path, steps=1, reuse_screen=True, solver=solver)
    assert generated == replay_weighted_chsh(tmp_path)
    assert generated['screen_statuses'] == {expected: 2}
    assert generated['verdicts']['disproved'] == 2
    assert calls == {'missing': [], 'init_error': ['build'], 'solve_error': ['build', 'solve', 'solve']}[policy]


def test_reusing_only_screen_preserves_all_exact_evidence_bytes(tmp_path):
    plain, reused = tmp_path / 'plain', tmp_path / 'template'
    assert run_weighted_chsh(plain, steps=4)['coverage_complete']
    assert run_weighted_chsh(reused, steps=4, reuse_screen=True)['coverage_complete']
    for k in range(5):
        relative = f'members/{k}.json'
        previous, new = (json.loads((d / relative).read_bytes()) for d in (plain, reused))
        assert previous['run_id'] != new['run_id']
        assert previous['screen']['status'] == new['screen']['status'] == 'completed'
        assert canonical_json(previous['assessment']) == canonical_json(new['assessment'])
        assert canonical_json(previous['evidence']) == canonical_json(new['evidence'])
