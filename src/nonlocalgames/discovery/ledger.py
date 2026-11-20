"""Cross-run evidence for the strict finite-quantum advantage over the local value.

The weighted CHSH manifests fix the two unrestricted resource contracts and a
zero gap threshold. A claim ID binds that query to the exact game verifier.
"""

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
import json
from math import isfinite
from operator import attrgetter
from os import PathLike
from pathlib import Path

from ..claims import evidence_from_dict
from ..claims.types import ClaimAssessment, EvidenceError, ExecutionStatus, TruthStatus, VerificationBudget
from ..core.scalar_wire import parse_rational, scalar_string
from . import _weighted_record
from ._checkpoint import MANIFEST_BYTES, MEMBER_BYTES, load_checkpoint, prefix_size
from .run_manifest import replay_budget, validate_manifest

__all__ = [
    'Ledger', 'Occurrence', 'QuestionRecord', 'QuestionStatus', 'RunArchive', 'VerifiedFact',
    'join_occurrences', 'ledger_to_dict', 'ledger_to_json', 'load_ledger', 'question_to_dict',
]


class QuestionStatus(StrEnum):
    """A resolution from completed replays, including disagreement and absent evidence."""

    PROVED = 'proved'
    DISPROVED = 'disproved'
    UNKNOWN = 'unknown'
    CONFLICT = 'conflict'
    UNVERIFIED = 'unverified'


@dataclass(frozen=True)
class RunArchive:
    """Identity and prefix coverage of one validated weighted CHSH manifest."""

    run_id: str
    family: str
    schema: str
    steps: int
    member_count: int

    @property
    def complete(self) -> bool:
        """Whether all scheduled members, including both endpoints, are present."""
        return self.member_count == self.steps + 1

    @property
    def missing_member_indices(self) -> tuple[int, ...]:
        """The unrecorded suffix after the contiguous prefix."""
        return tuple(range(self.member_count, self.steps + 1))


@dataclass(frozen=True)
class _MemberClaim:
    run_id: str
    member_id: str
    claim_id: str
    verifier_id: str

    @property
    def origin(self) -> tuple[str, str]:
        """Content-bound run and member identities, independent of filesystem location."""
        return self.run_id, self.member_id


@dataclass(frozen=True)
class Occurrence(_MemberClaim):
    """One archived observation. A numerical candidate and a receipt carry no proof.

    replay_execution is None when replay was skipped. A saved generation failure
    retains its execution status when reviewed and cannot produce a verified fact.
    """

    k: int
    p: Fraction
    archived_truth: TruthStatus
    archived_execution: ExecutionStatus
    screen_status: str
    candidate_gap: float | None
    replay_execution: ExecutionStatus | None = None

    def __post_init__(self):
        if not isinstance(self.p, Fraction):
            raise TypeError('A grid parameter must be an exact Fraction')
        if self.candidate_gap is not None and (
            type(self.candidate_gap) is not float or not isfinite(self.candidate_gap)
        ):
            raise ValueError('A numerical candidate gap must be a finite float')
        object.__setattr__(self, 'archived_truth', TruthStatus(self.archived_truth))
        object.__setattr__(self, 'archived_execution', ExecutionStatus(self.archived_execution))
        if self.replay_execution is not None:
            object.__setattr__(self, 'replay_execution', ExecutionStatus(self.replay_execution))


@dataclass(frozen=True)
class VerifiedFact(_MemberClaim):
    """The exact bounds, outcome and evidence references from one completed replay."""

    truth: TruthStatus
    execution: ExecutionStatus
    local_value: Fraction | None
    quantum_lower: Fraction | None
    quantum_upper: Fraction | None
    certified_gap_lower: Fraction | None
    verified_evidence: tuple[str, ...]
    numerical_hints: tuple[str, ...]
    derivation: tuple[str, ...]

    def __post_init__(self):
        if self.execution != ExecutionStatus.COMPLETED:
            raise ValueError('A verified fact requires a completed replay')
        object.__setattr__(self, 'truth', TruthStatus(self.truth))
        object.__setattr__(self, 'execution', ExecutionStatus(self.execution))
        for value in (self.local_value, self.quantum_lower, self.quantum_upper, self.certified_gap_lower):
            if value is not None and not isinstance(value, Fraction):
                raise TypeError('Verified bounds must be exact Fractions')
        for name in ('verified_evidence', 'numerical_hints', 'derivation'):
            ids = getattr(self, name)
            if not isinstance(ids, tuple) or any(not isinstance(i, str) for i in ids):
                raise TypeError(f'{name} must be a tuple of identifiers')
        object.__setattr__(self, 'verified_evidence', tuple(sorted(self.verified_evidence)))
        object.__setattr__(self, 'numerical_hints', tuple(sorted(self.numerical_hints)))


@dataclass(frozen=True)
class QuestionRecord:
    """One exact claim and its resolution across the supplied archives.

    Opposing decisive facts produce CONFLICT and both sides retain their origins.
    Incompatible verifier IDs also produce CONFLICT, with verifier_id set to None.
    verified_evidence cites only supporting facts. numerical_hints spans all facts.
    """

    claim_id: str
    verifier_id: str | None
    status: QuestionStatus
    reason: str
    occurrences: tuple[Occurrence, ...]
    facts: tuple[VerifiedFact, ...]
    supporting_facts: tuple[VerifiedFact, ...]
    conflicting_facts: tuple[VerifiedFact, ...]
    verified_evidence: tuple[str, ...]
    numerical_hints: tuple[str, ...]
    coverage_runs: tuple[str, ...]
    missing_runs: tuple[str, ...]


def _missing_runs(occurrences, covered, runs):
    # Validated manifests in this family bind the same grid convention and query.
    points = defaultdict(set)
    for occurrence in occurrences:
        points[runs[occurrence.run_id].family].add(occurrence.p)

    missing = []
    for run in runs.values():
        if run.complete or run.run_id in covered:
            continue
        for p in points[run.family]:
            k = p * run.steps
            if k.denominator == 1 and run.member_count <= k <= run.steps:
                missing.append(run.run_id)
                break
    return tuple(sorted(missing))


def _resolution(verifier_ids, facts):
    if len(verifier_ids) != 1:
        return QuestionStatus.CONFLICT, 'verifier_id_mismatch', (), facts
    proved = tuple(f for f in facts if f.truth == TruthStatus.PROVED)
    disproved = tuple(f for f in facts if f.truth == TruthStatus.DISPROVED)
    if proved and disproved:
        opposing = tuple(f for f in facts if f.truth != TruthStatus.UNKNOWN)
        return QuestionStatus.CONFLICT, 'opposing_verified_verdicts', (), opposing
    if proved:
        return QuestionStatus.PROVED, 'verified_proof', proved, ()
    if disproved:
        return QuestionStatus.DISPROVED, 'verified_disproof', disproved, ()
    if facts:
        return QuestionStatus.UNKNOWN, 'only_unknown_facts', (), ()
    return QuestionStatus.UNVERIFIED, 'no_verified_facts', (), ()


def join_occurrences(
    occurrences: Iterable[Occurrence], facts: Iterable[VerifiedFact], runs: Iterable[RunArchive] = (),
) -> tuple[QuestionRecord, ...]:
    """Join by claim_id. Callers supply already replayed facts or use load_ledger.

    coverage_runs lists every observing run. missing_runs counts only supplied
    runs of the same weighted family whose grid schedules an observed parameter
    in their missing suffix. Empty prefixes count, off-grid parameters do not.
    Without run descriptors the join makes no missing-coverage inference.
    """
    run_map = {}
    for run in runs:
        if run.run_id in run_map:
            raise ValueError(f'Duplicate run_id {run.run_id}')
        run_map[run.run_id] = run

    by_claim = defaultdict(dict)
    for occurrence in sorted(occurrences, key=attrgetter('origin')):
        observed = by_claim[occurrence.claim_id]
        if occurrence.origin in observed:
            raise ValueError(f'Duplicate occurrence at {occurrence.origin}')
        if run_map and occurrence.run_id not in run_map:
            raise ValueError(f'Unknown occurrence run_id {occurrence.run_id}')
        observed[occurrence.origin] = occurrence

    claim_facts = defaultdict(dict)
    for fact in sorted(facts, key=attrgetter('origin')):
        occurrence = by_claim.get(fact.claim_id, {}).get(fact.origin)
        if occurrence is None or occurrence.replay_execution != ExecutionStatus.COMPLETED:
            raise ValueError(f'Fact lacks a matching replayed occurrence at {fact.origin}')
        verified = claim_facts[fact.claim_id]
        if fact.origin in verified:
            raise ValueError(f'Duplicate fact at {fact.origin}')
        verified[fact.origin] = fact

    questions = []
    for claim_id in sorted(by_claim):
        observed = tuple(by_claim[claim_id].values())
        verified = tuple(claim_facts[claim_id].values())
        verifier_ids = {o.verifier_id for o in observed} | {f.verifier_id for f in verified}
        status, reason, support, conflicts = _resolution(verifier_ids, verified)
        covered = {o.run_id for o in observed}
        questions.append(QuestionRecord(
            claim_id=claim_id, verifier_id=next(iter(verifier_ids)) if len(verifier_ids) == 1 else None,
            status=status, reason=reason, occurrences=observed, facts=verified,
            supporting_facts=support, conflicting_facts=conflicts,
            verified_evidence=tuple(sorted({i for f in support for i in f.verified_evidence})),
            numerical_hints=tuple(sorted({i for f in verified for i in f.numerical_hints})),
            coverage_runs=tuple(sorted(covered)),
            missing_runs=_missing_runs(observed, covered, run_map) if run_map else (),
        ))
    return tuple(questions)


@dataclass(frozen=True)
class Ledger:
    """An immutable cross-run snapshot of archived observations and replayed facts."""

    runs: tuple[RunArchive, ...]
    occurrences: tuple[Occurrence, ...]
    facts: tuple[VerifiedFact, ...]
    questions: tuple[QuestionRecord, ...]

    def resolve(self, claim_id: str) -> QuestionRecord:
        """Look up an observed claim by exact identity. An unseen ID raises KeyError."""
        i = bisect_left(self.questions, claim_id, key=attrgetter('claim_id'))
        if i == len(self.questions) or self.questions[i].claim_id != claim_id:
            raise KeyError(claim_id)
        return self.questions[i]

    @property
    def summary(self) -> dict:
        """Count runs, origins, completed replays and every question status."""
        counts = Counter(q.status for q in self.questions)
        return {
            'runs': len(self.runs), 'questions': len(self.questions), 'occurrences': len(self.occurrences),
            'facts': len(self.facts), 'statuses': {s.value: counts[s] for s in QuestionStatus},
        }


def _origin_dict(member):
    return {'run_id': member.run_id, 'member_id': member.member_id}


def _occurrence_dict(o):
    return {
        **_origin_dict(o), 'claim_id': o.claim_id, 'verifier_id': o.verifier_id,
        'k': o.k, 'p': scalar_string(o.p),
        'archived_truth': o.archived_truth.value, 'archived_execution': o.archived_execution.value,
        'screen_status': o.screen_status, 'candidate_gap': o.candidate_gap,
        'replay_execution': o.replay_execution.value if o.replay_execution is not None else None,
    }


def _fact_dict(f):
    bounds = {}
    for name in ('local_value', 'quantum_lower', 'quantum_upper', 'certified_gap_lower'):
        value = getattr(f, name)
        bounds[name] = None if value is None else scalar_string(value)
    return {
        **_origin_dict(f), 'claim_id': f.claim_id, 'verifier_id': f.verifier_id,
        'truth': f.truth.value, 'execution': f.execution.value, **bounds,
        'verified_evidence': list(f.verified_evidence), 'numerical_hints': list(f.numerical_hints),
        'derivation': list(f.derivation),
    }


def question_to_dict(question: QuestionRecord) -> dict:
    """Encode one v1 resolution with exact bounds and origins on every piece of support."""
    return {
        'schema': 'nonlocalgames.ledger_question.v1', 'claim_id': question.claim_id,
        'verifier_id': question.verifier_id, 'status': question.status.value, 'reason': question.reason,
        'occurrences': [_occurrence_dict(o) for o in question.occurrences],
        'facts': [_fact_dict(f) for f in question.facts],
        'supporting_facts': [_origin_dict(f) for f in question.supporting_facts],
        'conflicting_facts': [_origin_dict(f) for f in question.conflicting_facts],
        'verified_evidence': list(question.verified_evidence), 'numerical_hints': list(question.numerical_hints),
        'coverage_runs': list(question.coverage_runs), 'missing_runs': list(question.missing_runs),
    }


def ledger_to_dict(ledger: Ledger) -> dict:
    """Encode a path-independent v1 ledger with occurrences and facts nested by claim."""
    return {
        'schema': 'nonlocalgames.evidence_ledger.v1', 'summary': ledger.summary,
        'runs': [
            {'run_id': r.run_id, 'family': r.family, 'schema': r.schema, 'steps': r.steps,
             'member_count': r.member_count, 'complete': r.complete,
             'missing_member_indices': list(r.missing_member_indices)}
            for r in ledger.runs
        ],
        'questions': [question_to_dict(q) for q in ledger.questions],
    }


def ledger_to_json(ledger: Ledger) -> str:
    """Serialize canonical ledger JSON, rejecting non-finite numbers."""
    return json.dumps(ledger_to_dict(ledger), sort_keys=True, separators=(',', ':'), allow_nan=False)


_ARCHIVE_ERRORS = (ValueError, OSError, TypeError, KeyError, IndexError, OverflowError)


def _archive_error(error, location):
    if isinstance(error, EvidenceError):
        return EvidenceError(error.issue.code, f'{location}: {error}', error.issue.evidence_id)
    return EvidenceError(ExecutionStatus.INVALID_EVIDENCE, f'{location}: {error}')


def _read_manifest(directory):
    location = f'archive {str(directory)!r}'
    try:
        if '..' in directory.parts:
            raise ValueError('Parent traversal in archive directory')
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError('Archive directory traverses a symlink')
        manifest = load_checkpoint(directory / 'manifest.json', MANIFEST_BYTES)
        if isinstance(manifest, dict) and isinstance(manifest.get('run_id'), str):
            location = f"run_id={manifest['run_id']}"
        manifest = validate_manifest(manifest)
        for entry in sorted(directory.iterdir()):
            if entry.name not in {'manifest.json', 'members'}:
                raise ValueError(f'Unexpected archive entry {entry.name!r}')
        return manifest
    except _ARCHIVE_ERRORS as error:
        raise _archive_error(error, location) from error


def _member_count(members, steps):
    if members.is_symlink():
        raise ValueError('Expected a members directory, found a symlink')
    try:
        count = prefix_size(members, steps)
    except ValueError as error:
        if members.is_dir():
            # Locate the grid position for a diagnostic only after prefix_size fails.
            indices = set()
            for entry in sorted(members.iterdir()):
                if entry.suffix != '.json' or not entry.stem.isascii() or not entry.stem.isdecimal():
                    continue
                k = int(entry.stem)
                if entry.name != f'{k}.json' or k > steps or entry.is_symlink() or not entry.is_file():
                    raise ValueError(f'k={k}: {error}') from error
                indices.add(k)
            for k in range(max(indices, default=-1)):
                if k not in indices:
                    raise ValueError(f'k={k}: {error}') from error
        raise
    # The resumable writer can ignore temporary files. The ledger imports a
    # recorded archive and requires every directory entry to be accounted for.
    if members.is_dir():
        names = {entry.name for entry in members.iterdir()}
        extra = names.difference(f'{k}.json' for k in range(count))
        if extra:
            raise ValueError(f'Unexpected member checkpoint {min(extra)!r}')
    return count


def _archived_verdict(record):
    if record['generation_error'] is not None:
        return TruthStatus.UNKNOWN, ExecutionStatus(record['generation_error']['status'])
    receipt = record['assessment']
    if (not isinstance(receipt, dict) or set(receipt) != set(ClaimAssessment.__dataclass_fields__)
            or receipt['schema'] != ClaimAssessment.schema
            or receipt['claim_id'] != record['claim_id'] or receipt['game_id'] != record['verifier_id']):
        raise ValueError('Invalid archived claim assessment fields, schema or binding')
    truth, execution = TruthStatus(receipt['truth']), ExecutionStatus(receipt['execution'])
    if execution != ExecutionStatus.COMPLETED and truth != TruthStatus.UNKNOWN:
        raise ValueError('A non-completed archived assessment cannot carry a verdict')
    return truth, execution


def _replayed_scalar(value):
    if value is None:
        return None
    # This canonical text was produced by the verifier from exact arithmetic.
    # Products and differences can exceed the scalar limits on artifact inputs.
    return parse_rational(value, canonical=True, max_chars=len(value), max_bits=4 * len(value))


def _load_observation(manifest, k, record, limit):
    result, review = None, None
    if limit is None:
        _weighted_record.validate_member_binding(manifest, k, record)
        _weighted_record.check_screen_shape(record['screen'], manifest)
        for raw in record['evidence']:
            artifact = evidence_from_dict(raw, max_bytes=MEMBER_BYTES)
            if artifact.game_id != record['verifier_id']:
                raise ValueError('Archived evidence belongs to a different game verifier')
    else:
        result, review = _weighted_record.verify_member_assessment(manifest, k, record, limit)
    truth, execution = _archived_verdict(record)
    occurrence = Occurrence(
        run_id=manifest['run_id'], member_id=record['member_id'], claim_id=record['claim_id'],
        verifier_id=record['verifier_id'], k=k, p=parse_rational(record['p'], canonical=True),
        archived_truth=truth, archived_execution=execution, screen_status=record['screen']['status'],
        candidate_gap=record['screen'].get('candidate_gap'),
        replay_execution=None if review is None else ExecutionStatus(review['execution']),
    )
    if result is None or result.execution != ExecutionStatus.COMPLETED:
        return occurrence, None
    fact = VerifiedFact(
        run_id=manifest['run_id'], member_id=record['member_id'], claim_id=result.claim_id,
        verifier_id=result.game_id, truth=result.truth, execution=result.execution,
        local_value=_replayed_scalar(review['local_value']),
        quantum_lower=_replayed_scalar(review['quantum_lower']),
        quantum_upper=_replayed_scalar(review['quantum_upper']),
        certified_gap_lower=_replayed_scalar(review['certified_gap_lower']),
        verified_evidence=result.verified_evidence, numerical_hints=result.numerical_hints,
        derivation=result.derivation,
    )
    return occurrence, fact


def _read_member(directory, manifest, k, limit):
    location = f"run_id={manifest['run_id']} k={k}"
    try:
        record = load_checkpoint(directory / 'members' / f'{k}.json', MEMBER_BYTES)
        if isinstance(record, dict) and isinstance(record.get('member_id'), str):
            location += f" member_id={record['member_id']}"
        return _load_observation(manifest, k, record, limit)
    except _ARCHIVE_ERRORS as error:
        raise _archive_error(error, location) from error


def load_ledger(
    paths: Iterable[str | PathLike[str]] | str | PathLike[str],
    replay: bool = True, budget: VerificationBudget | None = None,
) -> Ledger:
    """Load weighted CHSH archives and replay each observed member by default.

    With replay disabled, check checkpoint identities, evidence envelopes and
    receipt bindings, and keep the archived verdicts and screen observations.
    Only completed verifier assessments supply facts. Each archive's recorded
    verification limits bound the caller's budget. Read stable archive directories
    with a complete manifest and zero or more consecutive member checkpoints.
    """
    if not isinstance(replay, bool):
        raise TypeError('replay must be a Boolean')
    if budget is not None and not isinstance(budget, VerificationBudget):
        raise TypeError('Expected VerificationBudget for replay')
    if isinstance(paths, (str, PathLike)):
        paths = (paths,)
    archives = {}
    for path in paths:
        directory = Path(path)
        manifest = _read_manifest(directory)
        run_id = manifest['run_id']
        if run_id in archives:
            raise EvidenceError(ExecutionStatus.INVALID_EVIDENCE, f'Duplicate run_id={run_id}')
        archives[run_id] = (directory, manifest)

    runs, occurrences, facts = [], [], []
    for run_id, (directory, manifest) in sorted(archives.items()):
        try:
            count = _member_count(directory / 'members', manifest['steps'])
        except _ARCHIVE_ERRORS as error:
            raise _archive_error(error, f'run_id={run_id}') from error
        runs.append(RunArchive(run_id, manifest['family'], manifest['schema'], manifest['steps'], count))
        limit = replay_budget(manifest, budget) if replay else None
        for k in range(count):
            occurrence, fact = _read_member(directory, manifest, k, limit)
            occurrences.append(occurrence)
            if fact is not None:
                facts.append(fact)
    runs = tuple(runs)
    occurrences = tuple(sorted(occurrences, key=attrgetter('origin')))
    facts = tuple(sorted(facts, key=attrgetter('origin')))
    return Ledger(runs, occurrences, facts, join_occurrences(occurrences, facts, runs))
