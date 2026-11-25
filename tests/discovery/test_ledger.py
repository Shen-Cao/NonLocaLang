from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nonlocalgames.certificates.xor import verify_xor_dual
from nonlocalgames.claims import VerificationBudget, assessment_to_dict, evidence_from_dict, evidence_to_dict
from nonlocalgames.claims.serialization import assessment_from_dict
from nonlocalgames.claims.types import ClaimAssessment, ClaimIssue, EvidenceError, ExecutionStatus, TruthStatus
from nonlocalgames.core.identity import canonical_json
from nonlocalgames.core.scalar_wire import parse_rational
from nonlocalgames.discovery import _weighted_record as records
from nonlocalgames.discovery.ledger import (
    Ledger, Occurrence, RunArchive, VerifiedFact, join_occurrences, ledger_to_dict, ledger_to_json,
    load_ledger,
    question_to_dict,
)
from nonlocalgames.discovery.weighted_run import run_weighted_chsh
from nonlocalgames.games import biased_chsh


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
    with pytest.raises(ValueError, match='completed replay'):
        replace(fact, execution=ExecutionStatus.BUDGET_EXCEEDED)
    for occurrences, facts, message in (
        ((), (fact,), 'matching replayed occurrence'),
        ((replace(observation, replay_execution=None),), (fact,), 'matching replayed occurrence'),
        ((observation, observation), (fact,), 'Duplicate occurrence'),
        ((observation,), (fact, fact), 'Duplicate fact'),
    ):
        with pytest.raises(ValueError, match=message) as caught:
            join_occurrences(occurrences, facts)
        assert str(fact.origin) in str(caught.value)


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
    assert all(s in str(caught.value) for s in (run_id, 'k=1', _read(member)['member_id']))


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
    run_id = _read(path / 'manifest.json')['run_id']
    original = path / entry if entry != '.' else path
    target = tmp_path / 'link-target'
    original.rename(target)
    _link(original, target, directory=target.is_dir())
    with pytest.raises(EvidenceError) as caught:
        load_ledger([path], replay=False)
    message = str(caught.value)
    if entry.startswith('members'):
        assert run_id in message and ('members' if entry == 'members' else 'k=0') in message
    else:
        assert str(path) in message and ('symlink' if entry == '.' else 'manifest.json') in message
    assert caught.value.__cause__ is not None


def test_rejects_parent_traversal_in_archive_input(archives):
    fine, coarse = archives
    with pytest.raises(EvidenceError, match='Parent traversal'):
        load_ledger([coarse / '..' / fine.name], replay=False)


def _link(path, target, *, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f'Symlinks unavailable: {error}')


@pytest.mark.parametrize('replay', [True, False])
def test_permits_caller_selected_symlink_in_parent_of_archive_input(archives, tmp_path, replay):
    alias = tmp_path / 'alias'
    _link(alias, archives[0].parent, directory=True)
    assert load_ledger(alias / archives[0].name, replay=replay) == load_ledger(archives[0], replay=replay)


def test_dangling_members_directory_cannot_look_like_an_empty_prefix(archives, tmp_path):
    shutil.copyfile(archives[0] / 'manifest.json', tmp_path / 'manifest.json')
    _link(tmp_path / 'members', tmp_path / 'absent', directory=True)
    with pytest.raises(EvidenceError, match='real directory') as caught:
        load_ledger([tmp_path], replay=False)
    assert _read(tmp_path / 'manifest.json')['run_id'] in str(caught.value)


@pytest.mark.parametrize('entry_type', ['directory', 'symlink'])
def test_scratch_name_cannot_hide_other_file_types(archives, tmp_path, entry_type):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    entry = archive / 'members/.nlg-checkpoint-masked.tmp'
    if entry_type == 'directory':
        entry.mkdir()
    else:
        _link(entry, archive / 'members/0.json')
    with pytest.raises(EvidenceError) as caught:
        load_ledger([archive], replay=False)
    assert entry.name in str(caught.value) and _read(archive / 'manifest.json')['run_id'] in str(caught.value)


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


def test_archived_truth_must_agree_with_derived_gap(member_with_dual, reject_before_assess):
    def flip(row):
        receipt = row['assessment']
        assert receipt['truth'] == 'disproved'
        assert all(i['lower'] == i['upper'] == '1' for i in receipt['intervals'])
        receipt['truth'] = 'proved'

    _rewrite_member(member_with_dual, flip)
    reject_before_assess(member_with_dual, 'truth differs from the strict gap verdict')


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


@pytest.mark.parametrize('replay', [True, False])
@pytest.mark.parametrize('layout', ['missing', 'empty', 'symlink', 'file'])
def test_import_requires_real_members_directory(archives, tmp_path, layout, replay):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    members = archive / 'members'
    saved = members.rename(tmp_path / 'saved-members')
    manifest = _read(archive / 'manifest.json')
    if layout == 'empty':
        members.mkdir()
        ledger = load_ledger(archive, replay=replay)
        run, = ledger.runs
        assert run.run_id == manifest['run_id'] and run.member_count == 0 and not run.complete
        assert run.missing_member_indices == tuple(range(manifest['steps'] + 1))
        assert ledger.questions == ledger.occurrences == ledger.facts == ()
        return
    if layout == 'symlink':
        _link(members, saved, directory=True)
    elif layout == 'file':
        members.touch()
    with pytest.raises(EvidenceError, match='members') as caught:
        load_ledger(archive, replay=replay)
    assert manifest['run_id'] in str(caught.value) and caught.value.__cause__ is not None


def test_selected_file_is_reported_as_a_file(archives):
    path = archives[0] / 'manifest.json'
    with pytest.raises(EvidenceError, match='archive is a file') as caught:
        load_ledger(path)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize('at_root', [True, False])
def test_import_stops_at_first_extra_directory_entry(archives, tmp_path, monkeypatch, at_root):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    directory = archive if at_root else archive / 'members'
    allowed = {p.name for p in directory.iterdir()}
    for i in range(2048):
        (directory / f'.nlg-checkpoint-{i}.tmp').touch()
    real_scandir = os.scandir
    yielded = []

    @contextmanager
    def counted(path):
        with real_scandir(path) as entries:
            def count_entries():
                for entry in entries:
                    yielded.append(entry.name)
                    yield entry
            yield count_entries() if path == directory else entries

    monkeypatch.setattr(os, 'scandir', counted)
    with pytest.raises(EvidenceError) as caught:
        load_ledger(archive, replay=False)
    assert 1 <= len(yielded) <= len(allowed) + 1
    assert set(yielded[:-1]) <= allowed and yielded[-1] not in allowed
    assert yielded[-1] in str(caught.value) and _read(archive / 'manifest.json')['run_id'] in str(caught.value)


def test_stray_entry_is_not_attributed_to_a_missing_index(archives, tmp_path):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    (archive / 'members/1.json').unlink()
    (archive / 'members/stray').touch()
    with pytest.raises(EvidenceError, match='stray') as caught:
        load_ledger(archive)
    assert 'k=1' not in str(caught.value)


def _assert_rejected_origin(path, replay, match=None, *, budget=None):
    member = _read(path)
    with pytest.raises(EvidenceError, match=match) as caught:
        load_ledger(path.parent.parent, replay=replay, budget=budget)
    assert all(s in str(caught.value) for s in
               (member['run_id'], f"k={member['k']}", member['member_id']))
    assert caught.value.__cause__ is not None
    return caught.value


def _set_nested(record, path, value):
    for field in path[:-1]:
        record = record[field]
    record[path[-1]] = value


@pytest.mark.parametrize(('fields', 'value'), [
    (('truth',), 'uncertain'), (('execution',), 'timeout'),
    (('intervals',), {}), (('intervals', 0), {}), (('intervals', 0, 'resource'), 'foreign'),
    (('intervals', 0, 'lower'), 1), (('intervals', 0, 'upper'), '2/2'),
    (('intervals', 0, 'lower_refs'), None), (('intervals', 0, 'extra'), []),
    (('checks',), None), (('checks', 0), []), (('checks', 0, 'extra'), None),
    (('checks', 0, 'is_proof'), 1), (('checks', 0, 'checker'), {}), (('checks', 0, 'checker'), ''),
    (('checks', 0, 'resource'), False), (('checks', 0, 'attained_score'), '1.00'),
    (('checks', 0, 'lower'), 0.5), (('checks', 0, 'actual_dimensions'), '22'),
    (('checks', 0, 'actual_dimensions'), [True]), (('checks', 0, 'is_proof'), False),
    (('issues',), {}), (('issues',), [{'code': 'unknown', 'message': 'issue', 'evidence_id': None}]),
    (('issues',), [{'code': 'invalid_evidence', 'message': 17, 'evidence_id': None}]),
    (('issues',), [{'code': 'invalid_evidence', 'message': 'missing ID field'}]),
    (('derivation',), [2]), (('verified_evidence',), 'artifact'), (('numerical_hints',), {}),
])
def test_admission_rejects_malformed_nested_receipt(member_with_dual, reject_before_assess, fields, value):
    _rewrite_member(member_with_dual, lambda row: _set_nested(row['assessment'], fields, value))
    reject_before_assess(member_with_dual)


@pytest.mark.parametrize('fields', [
    ('verified_evidence', 0), ('numerical_hints', 0), ('checks', 0, 'artifact_id'),
    ('intervals', 0, 'lower_refs', 0), ('intervals', 0, 'upper_refs', 0), ('issues', 0, 'evidence_id'),
])
def test_fast_load_binds_each_receipt_reference_to_its_member(archives, tmp_path, fields):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    member = archive / 'members/0.json'
    foreign = _read(archive / 'members/1.json')['evidence'][0]['artifact_id']

    def change(row):
        receipt = row['assessment']
        if fields[0] == 'numerical_hints':
            receipt['numerical_hints'] = [foreign]
        elif fields[0] == 'issues':
            receipt['issues'] = [{'code': 'invalid_evidence', 'message': 'issue', 'evidence_id': foreign}]
        else:
            _set_nested(receipt, fields, foreign)

    _rewrite_member(member, change)
    _assert_rejected_origin(member, replay=False, match='evidence reference')


def test_fast_load_rejects_envelope_from_another_verifier(archives, tmp_path):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    member = archive / 'members/0.json'
    foreign = _read(archive / 'members/1.json')['evidence'][0]
    _rewrite_member(member, lambda row: _set_nested(row, ('evidence', 0), foreign))
    _assert_rejected_origin(member, replay=False, match='different game verifier')


def test_replay_validates_archived_observation_before_calling_assess_gap(archives, tmp_path, monkeypatch):
    archive = shutil.copytree(archives[0], tmp_path / 'archive')
    member = archive / 'members/0.json'
    _rewrite_member(member, lambda row: row['assessment'].update(checks=None))
    monkeypatch.setattr(records, 'assess_gap', lambda *a, **kw: pytest.fail('invalid receipt reached replay'))
    _assert_rejected_origin(member, replay=True, match='checks must be a JSON array')


@pytest.fixture
def member_with_dual(archives, tmp_path):
    archive = tmp_path / 'archive'
    (archive / 'members').mkdir(parents=True)
    shutil.copyfile(archives[0] / 'manifest.json', archive / 'manifest.json')
    member = archive / 'members/0.json'
    shutil.copyfile(archives[0] / 'members/0.json', member)
    return member


def _replace_payload(row, index, payload):
    """Make a new envelope and rebind every receipt reference before rehashing the member."""
    old = evidence_from_dict(row['evidence'][index])
    new = replace(old, payload_json=canonical_json(payload))
    row['evidence'][index] = evidence_to_dict(new)
    old_id, new_id = old.artifact_id, new.artifact_id
    receipt = row['assessment']
    assert receipt['checks'][index]['artifact_id'] == old_id
    receipt['checks'][index]['artifact_id'] = new_id
    refs = [receipt['verified_evidence']]
    for interval in receipt['intervals']:
        refs.extend((interval['lower_refs'], interval['upper_refs']))
    for ids in refs:
        if old_id in ids:
            ids[ids.index(old_id)] = new_id
            ids.sort()


@pytest.fixture(params=[False, True], ids=['fast', 'budget-limited'])
def reject_before_assess(request, monkeypatch):
    monkeypatch.setattr(records, 'assess_gap', lambda *a, **kw: pytest.fail('malformed receipt reached replay'))
    replay = request.param
    budget = VerificationBudget(max_strategies=1) if replay else None

    def reject(member, match=None):
        error = _assert_rejected_origin(member, replay=replay, match=match, budget=budget)
        assert error.issue.code == ExecutionStatus.INVALID_EVIDENCE
        return str(error.__cause__)

    return reject


@pytest.mark.parametrize('field', ['intervals', 'checks'])
def test_empty_completed_receipt_sections_reject_before_replay(member_with_dual, reject_before_assess, field):
    _rewrite_member(member_with_dual, lambda row: row['assessment'].update({field: []}))
    reject_before_assess(member_with_dual, field)


def test_completed_receipt_rejects_duplicate_verified_evidence(member_with_dual, reject_before_assess):
    def duplicate(row):
        ids = row['assessment']['verified_evidence']
        ids.append(ids[0])
    _rewrite_member(member_with_dual, duplicate)
    reject_before_assess(member_with_dual, 'verified_evidence')


def test_completed_receipt_binds_local_check_to_local_resource(member_with_dual, reject_before_assess):
    _rewrite_member(member_with_dual, lambda row: row['assessment']['checks'][0].update(resource='quantum_finite'))
    reject_before_assess(member_with_dual, 'local_enumeration')


def test_completed_receipt_requires_two_quantum_dimensions(member_with_dual, reject_before_assess):
    _rewrite_member(member_with_dual, lambda row: row['assessment']['checks'][1].update(actual_dimensions=[2]))
    reject_before_assess(member_with_dual, 'quantum_strategy')


@pytest.mark.parametrize('change', ['quantum-dimensions', 'quantum-lower', 'local-value', 'xor-upper'])
def test_only_completed_replay_compares_receipt_to_witness(member_with_dual, change):
    def corrupt(row):
        receipt = row['assessment']
        local_check, strategy, dual = receipt['checks']
        quantum, local = receipt['intervals']
        if change == 'quantum-dimensions':
            strategy['actual_dimensions'] = [2, 3]
        elif change == 'quantum-lower':
            strategy.update(lower='3/4', attained_score='3/4')
            quantum['lower_refs'].remove(strategy['artifact_id'])
        elif change == 'local-value':
            local_check.update(lower='3/4', upper='3/4', attained_score='3/4')
            quantum['lower_refs'].remove(local_check['artifact_id'])
            local.update(lower='3/4', upper='3/4', upper_refs=[local_check['artifact_id']])
            receipt['truth'] = 'proved'
        else:
            dual['upper'] = quantum['upper'] = '5/4'
            local['upper_refs'].remove(dual['artifact_id'])
            receipt['truth'] = 'unknown'

    _rewrite_member(member_with_dual, corrupt)
    archive = member_with_dual.parent.parent
    with patch.object(records, 'assess_gap', wraps=records.assess_gap) as assess:
        fast = load_ledger(archive, replay=False)
        occurrence, = fast.occurrences
        assert occurrence.replay_execution is None
        assert fast.facts == () and fast.questions[0].status == 'unverified'
        assess.assert_not_called()

        limited = load_ledger(archive, budget=VerificationBudget(max_strategies=1))
        occurrence, = limited.occurrences
        assert occurrence.replay_execution == 'budget_exceeded'
        assert limited.facts == () and limited.questions[0].status == 'unverified'
        assess.assert_called_once()

        assess.reset_mock()
        error = _assert_rejected_origin(member_with_dual, replay=True, match='Archived claim differs from replay')
        assert error.issue.code == ExecutionStatus.INVALID_EVIDENCE
        assess.assert_called_once()


@pytest.mark.parametrize(('payload_dims', 'check_dims'), [
    ([True, 2], [1, 2]), ([2.0, 2], [2, 2]), (['2', 2], [2, 2]), ([], [2, 2]),
    ([2], [2, 2]), ('22', [2, 2]), ([-1, 2], [1, 2]), ([0, 2], [1, 2]),
])
def test_quantum_payload_dimensions_require_two_positive_json_integers(
        member_with_dual, reject_before_assess, payload_dims, check_dims):
    def corrupt(row):
        payload = row['evidence'][1]['payload']
        strategy = {**payload['strategy'], 'dimensions': payload_dims}
        _replace_payload(row, 1, {**payload, 'strategy': strategy})
        row['assessment']['checks'][1]['actual_dimensions'] = check_dims

    _rewrite_member(member_with_dual, corrupt)
    reject_before_assess(member_with_dual, 'dimensions must be two positive integers')


@pytest.mark.parametrize('tables', [
    None, {'alice': [0, 0], 'bob': [0, 0]}, [[0, 0]], [[0], [0, 0]],
    [[False, 0], [0, 0]], [[0, 2], [0, 0]],
])
def test_malformed_local_response_tables(member_with_dual, reject_before_assess, tables):
    _rewrite_member(member_with_dual, lambda row: _replace_payload(
        row, 0, {**row['evidence'][0]['payload'], 'strategy': tables}))
    reject_before_assess(member_with_dual, '(?i)response tables')


@pytest.mark.parametrize(('change', 'value', 'message'), [
    ((), {'dimensions': [2, 2]}, 'fields or schema'),
    (('schema',), 'foreign', 'fields or schema'), (('extra',), None, 'fields or schema'),
    (('encoding',), 'float', 'encoding'), (('dimensions',), [2, 3], 'matrix shape'),
    (('state',), {'real': []}, 'real and imag'), (('state', 'real', 0, 0), '2/2', 'canonical'),
    (('measurements',), [], 'each party'), (('measurements', 0), [], 'per local input'),
    (('measurements', 0, 0), [], 'per local output'),
    (('measurements', 0, 0, 0, 'imag'), [[0]], 'matrix shape'),
    (('measurements', 0, 0, 0, 'real', 0, 0), 0, 'canonical rational strings'),
])
def test_malformed_quantum_strategy(member_with_dual, reject_before_assess, change, value, message):
    def corrupt(row):
        payload = deepcopy(row['evidence'][1]['payload'])
        if change:
            _set_nested(payload['strategy'], change, value)
        else:
            payload['strategy'] = value
        _replace_payload(row, 1, payload)
        if change == ('dimensions',):
            row['assessment']['checks'][1]['actual_dimensions'] = value

    _rewrite_member(member_with_dual, corrupt)
    reject_before_assess(member_with_dual, message)


def test_xor_objective_must_equal_declared_upper(member_with_dual, reject_before_assess):
    def corrupt(row):
        payload = deepcopy(row['evidence'][2]['payload'])
        payload['diagonal'][0] = '1/3'
        assert verify_xor_dual(biased_chsh(0), payload['diagonal']).value == Fraction(7, 6)
        assert payload['upper_bound'] == row['assessment']['checks'][2]['upper'] == '1'
        _replace_payload(row, 2, payload)

    _rewrite_member(member_with_dual, corrupt)
    reject_before_assess(member_with_dual, 'objective differs from its declared upper bound')


@pytest.mark.parametrize('kind', ['local', 'quantum', 'xor'])
def test_structurally_valid_false_witness_is_only_an_observation(member_with_dual, kind):
    def corrupt(row):
        index = ('local', 'quantum', 'xor').index(kind)
        payload = deepcopy(row['evidence'][index]['payload'])
        if kind == 'local':
            payload['strategy'] = [[0, 0], [1, 1]]
        elif kind == 'quantum':
            payload['strategy']['state']['real'][0][0] = '-1'
        else:
            # This diagonal has the declared objective and violates PSD.
            payload['diagonal'] = ['-1/6', '5/12', '1/6', '1/12']
        _replace_payload(row, index, payload)

    _rewrite_member(member_with_dual, corrupt)
    archive = member_with_dual.parent.parent
    fast = load_ledger(archive, replay=False)
    assert len(fast.occurrences) == 1 and fast.facts == ()
    replayed = load_ledger(archive)
    occurrence, = replayed.occurrences
    assert occurrence.replay_execution == 'invalid_evidence'
    assert replayed.facts == () and replayed.questions[0].status == 'unverified'


def test_wide_xor_uses_archive_budget_for_admission_and_caller_budget_for_replay(tmp_path):
    archive = tmp_path / 'wide'
    budget = VerificationBudget(max_wire_scalar_bits=8192)
    run_weighted_chsh(archive, steps=4, screen=False, verification_budget=budget, max_new=1)
    game = biased_chsh(0)
    delta = Fraction(1, 2**4100)

    def widen(row):
        diagonal = list(map(Fraction, row['evidence'][2]['payload']['diagonal']))
        diagonal[0] += delta
        # Both a payload component and the resulting receipt bound need over 4096 bits.
        dual = verify_xor_dual(game, diagonal, max_scalar_bits=budget.max_wire_scalar_bits)
        assert dual.value == 1 + delta and diagonal[0].denominator.bit_length() > 4096
        _replace_payload(row, 2, dual.witness)
        checked = records.assess_gap(game, records._CLAIM, tuple(map(evidence_from_dict, row['evidence'])),
                                     budget=budget)
        assert checked.execution == 'completed'
        row['assessment'] = assessment_to_dict(checked)

    _rewrite_member(archive / 'members/0.json', widen)
    fast = load_ledger(archive, replay=False)
    assert len(fast.occurrences) == 1 and fast.facts == () and fast.questions[0].status == 'unverified'
    full = load_ledger(archive, budget=budget)
    fact, = full.facts
    assert fact.quantum_upper == 1 + delta and fact.truth == 'unknown'
    for limit, artifact_index in ((VerificationBudget(), 2), (replace(budget, max_strategies=1), 0)):
        limited = load_ledger(archive, budget=limit)
        occurrence, = limited.occurrences
        assert limited.facts == () and limited.questions[0].status == 'unverified'
        issue, = occurrence.replay_issues
        assert occurrence.replay_execution == issue.code == 'budget_exceeded'
        assert issue.evidence_id == _read(archive / 'members/0.json')['evidence'][artifact_index]['artifact_id']


@pytest.mark.parametrize(('index', 'field', 'value', 'message'), [
    (0, 'value', 1, 'rational string'), (0, 'value', '2/2', 'canonical rational string'),
    (1, 'lower_bound', 1, 'rational string'), (1, 'lower_bound', '1.0', 'canonical rational string'),
    (2, 'upper_bound', 1, 'rational string'), (2, 'upper_bound', '+1', 'canonical rational string'),
    (2, 'diagonal', '0000', 'diagonal'), (2, 'diagonal', ['0'], 'diagonal'),
    (2, 'diagonal', ['0', 0, '0', '0'], 'rational string'),
    (2, 'diagonal', ['0', '0', '0', '0/2'], 'canonical rational string'),
    (1, 'strategy', [], 'strategy'), (1, 'resource', 'local', 'resource'),
    (1, 'game_fingerprint', 'foreign', 'game fingerprint'), (2, 'schema', 'foreign', 'schema'),
])
def test_weighted_payload_declarations_must_be_well_formed(
        member_with_dual, reject_before_assess, index, field, value, message):
    _rewrite_member(member_with_dual, lambda row: _replace_payload(
        row, index, {**row['evidence'][index]['payload'], field: value}))
    reject_before_assess(member_with_dual, message)


@pytest.mark.parametrize('index', [0, 1, 2], ids=['local', 'quantum', 'xor'])
@pytest.mark.parametrize('change', ['not-object', 'extra-field', 'missing-field'])
def test_weighted_payload_requires_exact_field_set(member_with_dual, reject_before_assess, index, change):
    def corrupt(row):
        payload = row['evidence'][index]['payload'].copy()
        if change == 'not-object':
            payload = []
        elif change == 'extra-field':
            payload['extra'] = 0
        else:
            del payload[('strategy', 'strategy', 'diagonal')[index]]
        _replace_payload(row, index, payload)

    _rewrite_member(member_with_dual, corrupt)
    reject_before_assess(member_with_dual, 'payload fields')


def test_matching_rectangular_quantum_strategy_passes_admission_and_replay(member_with_dual):
    from nonlocalgames.quantum.exact import ExactQuantumStrategy, strategy_to_dict
    from nonlocalgames.quantum.exact.algebra import identity, scale

    # Both parties always answer zero, attaining one at p=0 with a 2-by-3 product space.
    measurements = tuple(((identity(d), scale(identity(d), 0)),) * 2 for d in (2, 3))
    strategy = ExactQuantumStrategy(scale(identity(6), Fraction(1, 6)), measurements)

    def replace_strategy(row):
        _replace_payload(row, 1, {**row['evidence'][1]['payload'], 'strategy': strategy_to_dict(strategy)})
        row['assessment']['checks'][1]['actual_dimensions'] = [2, 3]

    _rewrite_member(member_with_dual, replace_strategy)
    archive = member_with_dual.parent.parent
    assert _at(load_ledger(archive, replay=False), Fraction(0)).status == 'unverified'
    fact, = _at(load_ledger(archive), Fraction(0)).facts
    assert fact.quantum_lower == 1 and fact.member_id == _read(member_with_dual)['member_id']


@pytest.mark.parametrize(('index', 'update'), [
    (0, {'attained_score': '0'}), (0, {'actual_dimensions': [2, 2]}),
    (0, {'lower': None, 'upper': None, 'attained_score': None}),
    (1, {'upper': '1'}), (1, {'attained_score': '0'}),
    (2, {'upper': None}), (2, {'attained_score': '1'}), (2, {'lower': '0'}),
    (2, {'is_proof': False, 'upper': None}), (2, {'checker': 'untrusted_observation.v1'}),
])
def test_completed_receipt_requires_weighted_checker_fields(member_with_dual, reject_before_assess, index, update):
    _rewrite_member(member_with_dual, lambda row: row['assessment']['checks'][index].update(update))
    checker = ('local_enumeration_replay.v1', 'rational_quantum_strategy.v1', 'rational_xor_dual.v1')[index]
    if 'checker' in update or 'is_proof' in update:
        expected = 'Weighted xor_dual evidence requires a rational_xor_dual.v1 proof check on quantum_commuting'
    else:
        expected = f'Invalid bounds or dimensions for the {checker} check'
    assert reject_before_assess(member_with_dual) == expected


@pytest.mark.parametrize('change', ['duplicate', 'swap_ids'])
def test_completed_receipt_pairs_each_artifact_with_its_checker(member_with_dual, reject_before_assess, change):
    def corrupt(row):
        checks = row['assessment']['checks']
        if change == 'duplicate':
            checks[1] = checks[0].copy()
        else:
            checks[0]['artifact_id'], checks[1]['artifact_id'] = checks[1]['artifact_id'], checks[0]['artifact_id']
    _rewrite_member(member_with_dual, corrupt)
    reject_before_assess(member_with_dual, 'Duplicate weighted check' if change == 'duplicate' else 'proof check on')


@pytest.mark.parametrize('field', ['numerical_hints', 'issues'])
def test_completed_receipt_requires_empty_hints_and_issues(member_with_dual, reject_before_assess, field):
    def corrupt(row):
        entry = row['evidence'][0]['artifact_id']
        if field == 'issues':
            entry = {'code': 'checker_error', 'message': 'reported issue', 'evidence_id': entry}
        row['assessment'][field].append(entry)
    _rewrite_member(member_with_dual, corrupt)
    reject_before_assess(member_with_dual, 'hints|issues')


@pytest.mark.parametrize(('field', 'value'), [
    ('lower', '0'), ('upper', None), ('lower_refs', []), ('upper_refs', []),
])
def test_completed_receipt_interval_matches_bounds_and_references(member_with_dual, reject_before_assess, field, value):
    _rewrite_member(member_with_dual, lambda row: row['assessment']['intervals'][0].update({field: value}))
    reject_before_assess(member_with_dual, 'intervals')


@pytest.mark.parametrize('field', ['intervals', 'derivation'])
def test_completed_receipt_requires_derived_interval_and_rule_order(member_with_dual, reject_before_assess, field):
    _rewrite_member(member_with_dual, lambda row: row['assessment'][field].reverse())
    reject_before_assess(member_with_dual, field)


def test_completed_receipt_rejects_contradictory_derived_bounds(member_with_dual, reject_before_assess):
    def corrupt(row):
        receipt = row['assessment']
        dual = receipt['checks'][2]
        dual['upper'] = '1/2'
        for interval in receipt['intervals']:
            interval.update(upper=dual['upper'], upper_refs=[dual['artifact_id']])
    _rewrite_member(member_with_dual, corrupt)
    reject_before_assess(member_with_dual, 'Inconsistent bounds')


def test_stored_weighted_assessment_requires_completed_execution(member_with_dual, reject_before_assess):
    _rewrite_member(member_with_dual, lambda row: row['assessment'].update(
        execution='budget_exceeded', truth='unknown'))
    reject_before_assess(member_with_dual, 'completed weighted receipt')


def test_legacy_verifier_still_reports_divergent_receipt_after_replay(member_with_dual):
    _rewrite_member(member_with_dual, lambda row: row['assessment'].update(checks=[]))
    manifest = _read(member_with_dual.parent.parent / 'manifest.json')
    with pytest.raises(ValueError, match='^Archived claim differs from replay at member 0$'):
        records.verify_member(manifest, 0, _read(member_with_dual), VerificationBudget())


def test_standard_receipt_round_trip_remains_an_observation(archives):
    row = _read(archives[0] / 'members/1.json')
    receipt = assessment_from_dict(row['assessment'], claim_id=row['claim_id'], game_id=row['verifier_id'],
                                   artifacts=map(evidence_from_dict, row['evidence']))
    assert isinstance(receipt.intervals[0].lower, Fraction)
    assert assessment_to_dict(receipt) == row['assessment']
    q = load_ledger(archives[0], replay=False).resolve(row['claim_id'])
    assert q.occurrences[0].archived_truth == receipt.truth == 'proved'
    assert q.status == 'unverified' and q.facts == ()


@pytest.mark.parametrize('replay', [True, False])
@pytest.mark.parametrize('generation_failure', [True, False])
def test_rehashed_invalid_screen_candidate(archives, tmp_path, replay, generation_failure):
    archive = shutil.copytree(archives[1], tmp_path / 'archive')
    member = archive / 'members/0.json'

    def change(row):
        if generation_failure:
            row.update(evidence=[], assessment=None, generation_error={'status': 'budget_exceeded', 'detail': 'saved'})
            row['screen']['candidate_gap'] = 0.125
        else:
            row['screen']['candidate_gap'] = True

    _rewrite_member(member, change)
    _assert_rejected_origin(member, replay=replay, match='candidate gap')


def test_real_budget_outcomes_keep_each_issue_and_origin(archives):
    archive = archives[0]
    budget = VerificationBudget(max_strategies=1)
    ledger = load_ledger(archive, budget=budget)
    assert len(ledger.occurrences) == _read(archive / 'manifest.json')['steps'] + 1
    assert len(ledger.questions) == ledger.summary['statuses']['unverified'] == len(ledger.occurrences)
    assert ledger.facts == ()
    for occurrence in ledger.occurrences:
        member = _read(archive / 'members' / f'{occurrence.k}.json')
        issue, = occurrence.replay_issues
        assert occurrence.origin == (member['run_id'], member['member_id'])
        assert occurrence.archived_truth == member['assessment']['truth']
        assert occurrence.archived_execution == 'completed'
        assert issue.code == occurrence.replay_execution == ExecutionStatus.BUDGET_EXCEEDED
        assert issue.message == 'Classical replay exceeds max_strategies'
        assert issue.evidence_id == member['evidence'][0]['artifact_id']
        q = ledger.resolve(occurrence.claim_id)
        assert q.facts == q.verified_evidence == ()
        assert question_to_dict(q)['occurrences'][0]['replay_issues'] == [
            {'code': 'budget_exceeded', 'message': issue.message, 'evidence_id': issue.evidence_id}]
    manifest = _read(archive / 'manifest.json')
    with pytest.raises(EvidenceError, match='^Replay budget exhausted at member 0$'):
        records.verify_member(manifest, 0, _read(archive / 'members/0.json'), budget)


@pytest.mark.parametrize('status', [s for s in ExecutionStatus if s not in (
    ExecutionStatus.COMPLETED, ExecutionStatus.BUDGET_EXCEEDED)])
def test_preserves_other_returned_execution_statuses(archives, monkeypatch, status):
    def incomplete(game, claim, artifacts, *, budget):
        issue = ClaimIssue(status, 'checker diagnostic', artifacts[0].artifact_id)
        return ClaimAssessment(records.claim_id(game, claim), records.verifier_id(game),
                               TruthStatus.UNKNOWN, status, issues=(issue,))
    monkeypatch.setattr(records, 'assess_gap', incomplete)
    ledger = load_ledger(archives[0])
    assert ledger.facts == () and all(q.status == 'unverified' for q in ledger.questions)
    assert all(o.replay_execution == status and o.replay_issues[0].code == status
               for o in ledger.occurrences)
    with pytest.raises(ValueError, match=f'^Archived member 0 has rejected evidence: {status.value}$'):
        records.verify_member(_read(archives[0] / 'manifest.json'), 0,
                              _read(archives[0] / 'members/0.json'), VerificationBudget())


def test_error_raised_by_assess_gap_keeps_original_issue_and_chain(archives, monkeypatch):
    row = _read(archives[0] / 'members/0.json')
    exc = EvidenceError(ExecutionStatus.CHECKER_ERROR, 'checker raised', row['evidence'][0]['artifact_id'])
    def raised(*args, **kwargs):
        raise exc
    monkeypatch.setattr(records, 'assess_gap', raised)
    with pytest.raises(EvidenceError, match='checker raised') as caught:
        load_ledger(archives[0])
    assert caught.value.__cause__ is exc
    assert caught.value.issue.evidence_id == exc.issue.evidence_id
    assert caught.value.issue.code == exc.issue.code
    assert all(x in str(caught.value) for x in (row['run_id'], 'k=0', row['member_id']))


def test_direct_constructors_freeze_and_order_sequences(archives):
    ledger = load_ledger(archives)
    q = _at(ledger, Fraction(0))
    fields = ('occurrences', 'facts', 'supporting_facts', 'conflicting_facts', 'verified_evidence',
              'numerical_hints', 'coverage_runs', 'missing_runs')
    restored = replace(q, **{f: list(reversed(getattr(q, f))) for f in fields})
    assert restored == q and hash(restored) == hash(q)
    reordered = replace(ledger, questions=ledger.questions[::-1], runs=list(reversed(ledger.runs)))
    assert reordered == ledger and hash(reordered) == hash(ledger)
    assert all(reordered.resolve(question.claim_id) == question for question in ledger.questions)


def test_supplied_run_must_cover_its_occurrence(archives):
    ledger = load_ledger(archives[0], replay=False)
    run, = ledger.runs
    empty = replace(run, member_count=0)
    with pytest.raises(ValueError, match='outside recorded grid prefix') as caught:
        join_occurrences(ledger.occurrences, (), (empty,))
    assert run.run_id in str(caught.value)
    with pytest.raises(ValueError, match='outside recorded grid prefix'):
        replace(ledger, runs=(empty,))
