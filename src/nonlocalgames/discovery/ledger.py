"""Cross-run questions about weighted CHSH games, with replayable evidence origins."""

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
import json
from math import isfinite
from operator import attrgetter

from ..claims.types import ExecutionStatus, TruthStatus
from ..core.scalar_wire import scalar_string


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

    by_claim = defaultdict(list)
    by_origin = {}
    for occurrence in sorted(occurrences, key=attrgetter('origin')):
        if occurrence.origin in by_origin:
            raise ValueError(f'Duplicate occurrence at {occurrence.origin}')
        if run_map and occurrence.run_id not in run_map:
            raise ValueError(f'Unknown occurrence run_id {occurrence.run_id}')
        by_origin[occurrence.origin] = occurrence
        by_claim[occurrence.claim_id].append(occurrence)

    claim_facts = defaultdict(list)
    seen = set()
    for fact in sorted(facts, key=attrgetter('origin')):
        occurrence = by_origin.get(fact.origin)
        if (occurrence is None or occurrence.claim_id != fact.claim_id
                or occurrence.replay_execution != ExecutionStatus.COMPLETED):
            raise ValueError(f'Fact lacks a matching replayed occurrence at {fact.origin}')
        if fact.origin in seen:
            raise ValueError(f'Duplicate fact at {fact.origin}')
        seen.add(fact.origin)
        claim_facts[fact.claim_id].append(fact)

    questions = []
    for claim_id in sorted(by_claim):
        observed = tuple(by_claim[claim_id])
        verified = tuple(claim_facts[claim_id])
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
