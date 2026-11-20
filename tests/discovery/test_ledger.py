from collections import Counter
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
import shutil
from types import SimpleNamespace

import pytest

from nonlocalgames.claims import VerificationBudget
from nonlocalgames.claims.types import EvidenceError, ExecutionStatus, TruthStatus
from nonlocalgames.core.identity import canonical_json
from nonlocalgames.core.scalar_wire import parse_rational
from nonlocalgames.discovery import _weighted_record as records
from nonlocalgames.discovery.ledger import (
    Ledger, Occurrence, RunArchive, VerifiedFact, join_occurrences, ledger_to_dict, ledger_to_json,
    load_ledger,
    question_to_dict,
)
from nonlocalgames.discovery.weighted_run import run_weighted_chsh


@pytest.fixture(scope='module')
def archives(tmp_path_factory):
    """Only the coarse numerical screen is synthetic. Both runs generate real raw proofs."""
    import cvxpy

    root = tmp_path_factory.mktemp('weighted-ledger')
    fine, coarse = root / 'fine', root / 'coarse'
    run_weighted_chsh(fine, steps=4, grid_denominator=2, screen=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy, 'installed_solvers', lambda: ['SCS'])
        patch.setattr(records, 'xor_quantum_value', lambda game, **kwargs: SimpleNamespace(
            value=0.91, status='optimal', diagnostics={'solver': 'SCS'}))
        run_weighted_chsh(coarse, steps=4, grid_denominator=1, screen=True)
    return fine, coarse


def _read(path):
    return json.loads(path.read_bytes())


def _rewrite_member(path, update):
    record = _read(path)
    del record['member_id']
    update(record)
    digest = sha256(canonical_json(record).encode()).hexdigest()
    record['member_id'] = record['schema'] + ':' + digest
    path.write_text(json.dumps(record))


def _at(ledger, p):
    claim_id = next(o.claim_id for o in ledger.occurrences if o.p == p)
    return ledger.resolve(claim_id)


def _origins(entries):
    return [(e.run_id, e.member_id) for e in entries]


def test_replays_every_member_and_resolves_from_verified_support(archives, monkeypatch):
    fine, coarse = archives
    calls = []
    assess = records.assess_gap

    def count_assessment(*args, **kwargs):
        result = assess(*args, **kwargs)
        calls.append(result.claim_id)
        return result

    monkeypatch.setattr(records, 'assess_gap', count_assessment)
    ledger = load_ledger(archives)
    assert len(ledger.runs) == 2
    assert len(ledger.questions) == 5
    assert len(ledger.facts) == len(ledger.occurrences) == 10
    assert Counter(calls) == Counter(o.claim_id for o in ledger.occurrences)
    fine_id = _read(fine / 'manifest.json')['run_id']
    coarse_id = _read(coarse / 'manifest.json')['run_id']
    question = _at(ledger, Fraction(1, 4))
    assert question.status == 'proved'
    assert question.coverage_runs == tuple(sorted((coarse_id, fine_id)))
    assert question.missing_runs == ()
    assert {f.run_id: f.truth for f in question.facts} == {
        fine_id: TruthStatus.PROVED, coarse_id: TruthStatus.UNKNOWN}
    assert _origins(question.facts) == _origins(question.occurrences)
    fine_fact, = question.supporting_facts
    assert fine_fact.run_id == fine_id
    receipt = _read(fine / 'members/1.json')['assessment']
    assert fine_fact.verified_evidence == tuple(sorted(receipt['verified_evidence']))
    assert fine_fact.numerical_hints == tuple(sorted(receipt['numerical_hints']))
    assert sorted(fine_fact.derivation) == sorted(receipt['derivation'])
    assert question.conflicting_facts == ()
    assert question.verified_evidence == tuple(sorted(fine_fact.verified_evidence))
    coarse_obs = next(o for o in question.occurrences if o.run_id == coarse_id)
    coarse_fact = next(f for f in question.facts if f.run_id == coarse_id)
    assert set(coarse_fact.verified_evidence) - set(question.verified_evidence)
    assert coarse_obs.archived_truth == 'unknown' and coarse_obs.screen_status == 'completed'
    assert coarse_obs.candidate_gap == 0.91 - 0.75
    assert str(coarse_obs.candidate_gap) not in question.verified_evidence
    for occurrence in question.occurrences:
        assert occurrence.archived_execution == occurrence.replay_execution == 'completed'
    zero = _at(ledger, Fraction(0))
    assert zero.status == 'disproved' and zero.supporting_facts == zero.facts
    assert len(zero.supporting_facts) == 2
    assert all(isinstance(f.quantum_upper, Fraction) and f.quantum_upper == 1 for f in zero.facts)
    assert zero.verified_evidence == tuple(sorted({
        evidence for fact in zero.facts for evidence in fact.verified_evidence}))


def test_observation_only_never_replays_or_promotes_receipts(archives, monkeypatch):
    monkeypatch.setattr(records, 'assess_gap', lambda *a, **kw: pytest.fail('unexpected replay'))
    ledger = load_ledger(archives, replay=False)
    assert ledger.facts == ()
    assert len(ledger.questions) == 5
    assert len(ledger.occurrences) == 10
    assert any(o.archived_truth == 'proved' for o in ledger.occurrences)
    assert all(o.replay_execution is None for o in ledger.occurrences)
    for question in ledger.questions:
        assert ledger.resolve(question.claim_id).status == 'unverified'
        assert question.facts == question.supporting_facts == question.verified_evidence == ()


def _synthetic(run_id, truth, *, claim_id='claim', verifier_id='verifier'):
    identity = dict(claim_id=claim_id, verifier_id=verifier_id, run_id=run_id,
                    member_id='member-' + run_id)
    completed = ExecutionStatus.COMPLETED
    occurrence = Occurrence(**identity, k=1, p=Fraction(1, 4), archived_truth=truth,
                            archived_execution=completed, replay_execution=completed,
                            screen_status='disabled', candidate_gap=None)
    local = Fraction(1, 2)
    lower = Fraction(3, 4) if truth == TruthStatus.PROVED else local
    upper = local if truth == TruthStatus.DISPROVED else None
    fact = VerifiedFact(**identity, truth=truth, execution=completed,
                        local_value=local, quantum_lower=lower, quantum_upper=upper,
                        certified_gap_lower=lower-local, verified_evidence=('proof-' + run_id,),
                        numerical_hints=('hint-' + run_id,), derivation=('rule',))
    return occurrence, fact


def test_pure_join_preserves_conflicts_support_and_unknowns():
    a, yes = _synthetic('a', TruthStatus.PROVED)
    b, no = _synthetic('b', TruthStatus.DISPROVED)
    c, unknown = _synthetic('c', TruthStatus.UNKNOWN)
    conflict, = join_occurrences((c, b, a), (no, unknown, yes))
    assert conflict.status == 'conflict'
    assert conflict.facts == (yes, no, unknown)
    assert conflict.occurrences == (a, b, c)
    assert conflict.conflicting_facts == (yes, no)
    assert conflict.supporting_facts == conflict.verified_evidence == ()
    assert conflict.numerical_hints == ('hint-a', 'hint-b', 'hint-c')
    for fact, occurrence in ((yes, a), (no, b)):
        resolved, = join_occurrences((c, occurrence), (unknown, fact))
        assert resolved.status == fact.truth
        assert resolved.facts == (fact, unknown)
        assert resolved.supporting_facts == (fact,)
        assert resolved.verified_evidence == fact.verified_evidence
        assert resolved.numerical_hints == (*fact.numerical_hints, 'hint-c')
    unresolved, = join_occurrences((c,), (unknown,))
    assert unresolved.status == 'unknown'
    assert unresolved.facts == (unknown,)
    assert unresolved.supporting_facts == unresolved.verified_evidence == ()
    pending, = join_occurrences((replace(c, replay_execution=None),), ())
    assert pending.status == 'unverified' and pending.facts == ()


def test_pure_join_requires_a_distinct_completed_origin_for_each_fact():
    observation, fact = _synthetic('a', TruthStatus.UNKNOWN)
    with pytest.raises(ValueError):
        replace(fact, execution=ExecutionStatus.BUDGET_EXCEEDED)
    with pytest.raises(ValueError):
        join_occurrences((), (fact,))
    with pytest.raises(ValueError):
        join_occurrences((replace(observation, replay_execution=None),), (fact,))
    with pytest.raises(ValueError):
        join_occurrences((observation, observation), (fact,))
    with pytest.raises(ValueError):
        join_occurrences((observation,), (fact, fact))


def test_equal_parameters_do_not_merge_distinct_claims_or_hide_provenance_conflicts():
    a, proof = _synthetic('a', TruthStatus.PROVED)
    b, other = _synthetic('b', TruthStatus.PROVED, claim_id='other')
    questions = join_occurrences((b, a), (other, proof))
    assert [q.claim_id for q in questions] == ['claim', 'other']
    same_member = replace(a, claim_id='second-question')
    questions = join_occurrences((same_member, a), (proof, replace(proof, claim_id='second-question')))
    assert [q.claim_id for q in questions] == ['claim', 'second-question']
    assert all(q.supporting_facts[0].origin == a.origin for q in questions)
    c, mismatch = _synthetic('c', TruthStatus.PROVED, verifier_id='different')
    conflict, = join_occurrences((c, a), (mismatch, proof))
    assert conflict.status == 'conflict' and conflict.verifier_id is None
    assert conflict.reason == 'verifier_id_mismatch'
    assert conflict.conflicting_facts == (proof, mismatch)
    assert conflict.supporting_facts == conflict.verified_evidence == ()
    inconsistent = replace(proof, verifier_id='different')
    within_origin, = join_occurrences((a,), (inconsistent,))
    assert within_origin.status == 'conflict'
    assert within_origin.reason == 'verifier_id_mismatch'


def test_partial_and_empty_prefix_only_miss_questions_on_their_own_grid(archives, tmp_path):
    partial, empty = tmp_path / 'partial', tmp_path / 'empty'
    run_weighted_chsh(partial, steps=4, grid_denominator=3, screen=False, max_new=2)
    run_weighted_chsh(empty, steps=2, grid_denominator=3, screen=False, max_new=0)
    partial_id = _read(partial / 'manifest.json')['run_id']
    empty_id = _read(empty / 'manifest.json')['run_id']
    fine_id = _read(archives[0] / 'manifest.json')['run_id']
    ledger = load_ledger([partial, empty, archives[0]])
    runs = {r.run_id: r for r in ledger.runs}
    assert runs[fine_id].complete and runs[fine_id].missing_member_indices == ()
    assert not runs[partial_id].complete and runs[partial_id].member_count == 2
    assert runs[partial_id].missing_member_indices == (2, 3, 4)
    assert not runs[empty_id].complete and runs[empty_id].member_count == 0
    assert runs[empty_id].missing_member_indices == (0, 1, 2)
    assert sorted(o.k for o in ledger.occurrences if o.run_id == partial_id) == [0, 1]
    assert len(ledger.facts) == len(ledger.occurrences) == 7
    assert _at(ledger, Fraction(1, 4)).missing_runs == ()
    q = _at(ledger, Fraction(1, 2))
    assert q.coverage_runs == (fine_id,)
    assert q.missing_runs == tuple(sorted((partial_id, empty_id)))
    assert _at(ledger, Fraction(3, 4)).missing_runs == (partial_id,)


def test_generation_failure_records_replay_execution_without_creating_a_fact(tmp_path):
    run_weighted_chsh(tmp_path, steps=4, screen=False, max_new=1,
                      verification_budget=VerificationBudget(max_strategies=1))
    ledger = load_ledger([tmp_path])
    q, = ledger.questions
    occurrence, = q.occurrences
    assert occurrence.archived_truth == TruthStatus.UNKNOWN
    assert occurrence.archived_execution == occurrence.replay_execution == ExecutionStatus.BUDGET_EXCEEDED
    assert q.status == 'unverified' and q.facts == ledger.facts == ()


def test_replay_budget_error_keeps_cause_and_origin(archives):
    member = _read(archives[0] / 'members/0.json')
    with pytest.raises(EvidenceError) as caught:
        load_ledger(archives[:1], budget=VerificationBudget(max_artifacts=1))
    assert caught.value.issue.code == ExecutionStatus.BUDGET_EXCEEDED
    assert isinstance(caught.value.__cause__, EvidenceError)
    assert all(x in str(caught.value) for x in (member['run_id'], 'k=0', member['member_id']))


def test_duplicate_run_id_is_rejected_even_at_another_directory(archives, tmp_path):
    copy = shutil.copytree(archives[0], tmp_path / 'other')
    run_id = _read(copy / 'manifest.json')['run_id']
    with pytest.raises(EvidenceError) as caught:
        load_ledger([copy, archives[0]], replay=False)
    assert run_id in str(caught.value)


def test_rejects_missing_and_malformed_manifest(archives, tmp_path):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    path = archive / 'manifest.json'
    manifest = _read(path)
    path.unlink()
    with pytest.raises(EvidenceError, match='manifest'):
        load_ledger([archive], replay=False)
    path.write_text(json.dumps(dict(manifest, steps=True)))
    with pytest.raises(EvidenceError) as caught:
        load_ledger([archive], replay=False)
    assert manifest['run_id'] in str(caught.value)


@pytest.mark.parametrize('extra', ['stray', 'members/stray', 'members/01.json'])
def test_rejects_extra_and_noncanonical_files(archives, tmp_path, extra):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    run_id = _read(archive / 'manifest.json')['run_id']
    (archive / extra).write_text('{}')
    with pytest.raises(EvidenceError) as caught:
        load_ledger([archive], replay=False)
    assert run_id in str(caught.value)


def test_middle_hole_reports_run_and_missing_index(archives, tmp_path):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    run_id = _read(archive / 'manifest.json')['run_id']
    (archive / 'members/1.json').unlink()
    with pytest.raises(EvidenceError) as caught:
        load_ledger([archive], replay=False)
    assert run_id in str(caught.value) and 'k=1' in str(caught.value)


@pytest.mark.parametrize('update', [{'run_id': 'foreign'}, {'claim_id': 'different'},
                                    {'verifier_id': 'different'}, {'k': 0}, {'p': '3/4'}])
def test_rejects_rehashed_member_with_wrong_binding_without_replay(archives, tmp_path, update):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    member = archive / 'members/1.json'
    run_id = _read(member)['run_id']
    _rewrite_member(member, lambda r: r.update(update))
    with pytest.raises(EvidenceError) as caught:
        load_ledger([archive], replay=False)
    assert run_id in str(caught.value) and 'k=1' in str(caught.value)


@pytest.mark.parametrize('damage', ['stale_member_id', 'nonfinite_candidate'])
def test_corrupt_members_fail_even_without_replay(archives, tmp_path, damage):
    archive = shutil.copytree(archives[1], tmp_path / 'archive')
    member = archive / 'members/1.json'
    receipt = _read(member)
    if damage == 'stale_member_id':
        receipt['member_id'] = 'stale'
    else:
        receipt['screen']['candidate_gap'] = float('inf')
    member.write_text(json.dumps(receipt))
    with pytest.raises(EvidenceError) as caught:
        load_ledger([archive], replay=False)
    assert receipt['run_id'] in str(caught.value) and 'k=1' in str(caught.value)


@pytest.mark.parametrize('entry', ['manifest.json', 'members/0.json', 'members', '.'])
def test_refuses_symlinked_checkpoints_and_archive_directories(archives, tmp_path, entry):
    path = shutil.copytree(archives[0], tmp_path / 'archive')
    original = path / entry if entry != '.' else path
    target = tmp_path / 'link-target'
    original.rename(target)
    _link(original, target, directory=target.is_dir())
    with pytest.raises(EvidenceError):
        load_ledger([path], replay=False)


def test_rejects_parent_traversal_in_archive_input(archives):
    fine, coarse = archives
    with pytest.raises(EvidenceError):
        load_ledger([coarse / '..' / fine.name], replay=False)


def _link(path, target, *, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f'Symlinks unavailable: {error}')


def test_rejects_symlink_in_parent_of_archive_input(archives, tmp_path):
    alias = tmp_path / 'alias'
    _link(alias, archives[0].parent, directory=True)
    with pytest.raises(EvidenceError):
        load_ledger([alias / archives[0].name], replay=False)


def test_dangling_members_directory_cannot_look_like_an_empty_prefix(archives, tmp_path):
    shutil.copyfile(archives[0] / 'manifest.json', tmp_path / 'manifest.json')
    _link(tmp_path / 'members', tmp_path / 'absent', directory=True)
    with pytest.raises(EvidenceError):
        load_ledger([tmp_path], replay=False)


@pytest.mark.parametrize('entry_type', ['directory', 'symlink'])
def test_scratch_name_cannot_hide_other_file_types(archives, tmp_path, entry_type):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    entry = archive / 'members/.nlg-checkpoint-masked.tmp'
    if entry_type == 'directory':
        entry.mkdir()
    else:
        _link(entry, archive / 'members/0.json')
    with pytest.raises(EvidenceError):
        load_ledger([archive], replay=False)


@pytest.mark.parametrize('field', ['claim_id', 'game_id'])
def test_fast_load_binds_embedded_archived_receipt_to_member(archives, tmp_path, field):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    member = archive / 'members/1.json'
    run_id = _read(member)['run_id']
    _rewrite_member(member, lambda r: r['assessment'].update({field: 'foreign'}))
    with pytest.raises(EvidenceError) as caught:
        load_ledger([archive], replay=False)
    assert run_id in str(caught.value) and 'k=1' in str(caught.value)


def test_fast_load_checks_raw_evidence_envelopes(archives, tmp_path):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    member = archive / 'members/1.json'
    _rewrite_member(member, lambda r: r['evidence'][0].pop('artifact_id'))
    damaged = _read(member)
    with pytest.raises(EvidenceError) as caught:
        load_ledger(archive, replay=False)
    assert all(x in str(caught.value) for x in (damaged['run_id'], 'k=1', damaged['member_id']))


def test_rehashed_archived_verdict_fails_replay_with_member_context(archives, tmp_path):
    path = shutil.copytree(archives[0], tmp_path / 'archive')
    member = path / 'members/0.json'
    _rewrite_member(member, lambda r: r['assessment'].update(truth='proved'))
    forged = _read(member)
    fast = load_ledger(path, replay=False).resolve(forged['claim_id'])
    assert fast.occurrences[0].archived_truth == 'proved'
    assert fast.status == 'unverified' and fast.facts == fast.verified_evidence == ()
    with pytest.raises(EvidenceError) as caught:
        load_ledger([path])
    message = str(caught.value)
    assert all(x in message for x in ('k=0', forged['run_id'], forged['member_id']))
    assert 'replay' in message.lower()


def test_wire_values_are_exact_and_serialization_is_path_and_input_order_independent(archives):
    ledger = load_ledger(archives)
    expected = ledger_to_json(ledger).encode()
    for paths in (tuple(reversed(archives)), archives):
        again = load_ledger(paths)
        assert again.questions == ledger.questions
        assert ledger_to_json(again).encode() == expected
    assert [r.run_id for r in ledger.runs] == sorted(r.run_id for r in ledger.runs)
    assert [q.claim_id for q in ledger.questions] == sorted(q.claim_id for q in ledger.questions)
    assert _origins(ledger.occurrences) == sorted(_origins(ledger.occurrences))
    assert _origins(ledger.facts) == sorted(_origins(ledger.facts))
    data = ledger_to_dict(ledger)
    assert expected == json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert str(archives[0].parent).encode() not in expected
    assert data['schema'] == 'nonlocalgames.evidence_ledger.v1'
    q = _at(ledger, Fraction(1, 4))
    wire = question_to_dict(q)
    assert wire['schema'] == 'nonlocalgames.ledger_question.v1'
    assert wire in data['questions']
    fact, = q.supporting_facts
    assert fact.local_value == Fraction(3, 4)
    assert fact.quantum_lower == Fraction(421, 500)
    assert fact.certified_gap_lower == Fraction(23, 250)
    as_dict = next(item for item in wire['facts'] if item['run_id'] == fact.run_id)
    for name in ('local_value', 'quantum_lower', 'quantum_upper', 'certified_gap_lower'):
        value = getattr(fact, name)
        if value is None:
            assert as_dict[name] is None
        else:
            assert isinstance(value, Fraction) and isinstance(as_dict[name], str)
            assert parse_rational(as_dict[name], canonical=True) == value
    assert as_dict['verified_evidence'] == list(fact.verified_evidence)
    assert as_dict['derivation'] == list(fact.derivation)
    assert wire['verified_evidence'] == list(q.verified_evidence)


def test_empty_ledger_wire_bytes_are_versioned():
    expected = (
        b'{"questions":[],"runs":[],"schema":"nonlocalgames.evidence_ledger.v1",'
        b'"summary":{"facts":0,"occurrences":0,"questions":0,"runs":0,'
        b'"statuses":{"conflict":0,"disproved":0,"proved":0,"unknown":0,"unverified":0}}}'
    )
    assert ledger_to_json(load_ledger([])).encode() == expected


def test_fixed_nonempty_ledger_has_exact_wire_record():
    obs, fact = _synthetic('a', TruthStatus.PROVED)
    obs = replace(obs, k=0, p=Fraction(0))
    run = RunArchive(run_id='a', family='family', schema='run-schema', steps=1, member_count=1)
    questions = join_occurrences((obs,), (fact,), (run,))
    ledger = Ledger(runs=(run,), occurrences=(obs,), facts=(fact,), questions=questions)

    origin = {'run_id': 'a', 'member_id': 'member-a'}
    binding = {**origin, 'claim_id': 'claim', 'verifier_id': 'verifier'}
    expected = {
        'schema': 'nonlocalgames.evidence_ledger.v1',
        'summary': {'runs': 1, 'questions': 1, 'occurrences': 1, 'facts': 1,
                    'statuses': {'proved': 1, 'disproved': 0, 'unknown': 0,
                                 'conflict': 0, 'unverified': 0}},
        'runs': [{'run_id': 'a', 'schema': 'run-schema', 'family': 'family', 'steps': 1,
                  'member_count': 1, 'complete': False, 'missing_member_indices': [1]}],
        'questions': [{
            'schema': 'nonlocalgames.ledger_question.v1', 'claim_id': 'claim',
            'verifier_id': 'verifier', 'status': 'proved', 'reason': 'verified_proof',
            'coverage_runs': ['a'], 'missing_runs': [], 'supporting_facts': [origin],
            'conflicting_facts': [], 'verified_evidence': ['proof-a'], 'numerical_hints': ['hint-a'],
            'occurrences': [{**binding, 'k': 0, 'p': '0', 'archived_truth': 'proved',
                             'archived_execution': 'completed', 'screen_status': 'disabled',
                             'candidate_gap': None, 'replay_execution': 'completed'}],
            'facts': [{**binding, 'truth': 'proved', 'execution': 'completed', 'local_value': '1/2',
                       'quantum_lower': '3/4', 'quantum_upper': None, 'certified_gap_lower': '1/4',
                       'verified_evidence': ['proof-a'], 'numerical_hints': ['hint-a'],
                       'derivation': ['rule']}],
        }],
    }
    assert ledger_to_json(ledger).encode() == json.dumps(
        expected, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
