"""Shared raw-evidence admission and consistency checking for claim families."""
from dataclasses import dataclass
from itertools import islice

from ..core.game import Game
from ..core.identity import verifier_id
from .resources import STANDARD_CHAIN, includes
from .types import (CertifiedInterval, ClaimIssue, EvidenceCheck, EvidenceError,
                    ExecutionStatus, VerificationBudget)
from .verify import verify_artifact


@dataclass(frozen=True)
class EvidenceBundle:
    """A verification receipt, never an input proof token for public assessment APIs."""
    game_id: str
    execution: ExecutionStatus
    intervals: tuple[CertifiedInterval, ...] = ()
    checks: tuple[EvidenceCheck, ...] = ()
    verified_evidence: tuple[str, ...] = ()
    numerical_hints: tuple[str, ...] = ()
    derivation: tuple[str, ...] = ()
    issues: tuple[ClaimIssue, ...] = ()


def bound_intervals(checks):
    """Close exact bounds under standard inclusions; attainment also supplies a lower bound."""
    intervals, rules = [], set()
    for target in STANDARD_CHAIN:
        lowers, uppers = [], []
        for check in checks:
            if not check.is_proof:
                continue
            if includes(target, check.resource):
                candidates = tuple(v for v in (check.lower, check.attained_score) if v is not None)
                if candidates:
                    lowers.append((max(candidates), check.artifact_id))
                    rules.add(f'lower:{check.resource.value}->{target.value}')
            if check.upper is not None and includes(check.resource, target):
                uppers.append((check.upper, check.artifact_id))
                rules.add(f'upper:{check.resource.value}->{target.value}')
        lower = max((v for v, _ in lowers), default=None)
        upper = min((v for v, _ in uppers), default=None)
        intervals.append(CertifiedInterval(target, lower, upper,
            tuple(sorted({ref for v, ref in lowers if v == lower})),
            tuple(sorted({ref for v, ref in uppers if v == upper}))))
    return tuple(intervals), tuple(sorted(rules))


def verify_evidence_bundle(game, artifacts=(), *, budget=None):
    """Reverify raw artifacts once, then expose bounds and distinct attainment facts.

    This function does not decide a mathematical claim. All public claim APIs
    accept raw artifacts and invoke admission themselves; a supplied bundle or
    checker-output object cannot bypass verification. Count/size limits do not
    include a wall-clock promise for verifier-identity construction.
    """
    if not isinstance(game, Game):
        raise TypeError('Expected Game')
    budget = VerificationBudget() if budget is None else budget
    if not isinstance(budget, VerificationBudget):
        raise TypeError('Expected VerificationBudget')
    identity = verifier_id(game)
    def failure(code, message, evidence_id=None):
        return EvidenceBundle(identity, code, issues=(ClaimIssue(code, message, evidence_id),))
    try:
        artifacts = tuple(islice(iter(artifacts), budget.max_artifacts+1))
    except TypeError:
        return failure(ExecutionStatus.INVALID_EVIDENCE, 'Evidence must be an iterable of data artifacts')
    if len(artifacts) > budget.max_artifacts:
        return failure(ExecutionStatus.BUDGET_EXCEEDED, 'Evidence list exceeds max_artifacts')
    checks = []
    for artifact in artifacts:
        try:
            checks.append(verify_artifact(game, artifact, budget=budget))
        except EvidenceError as error:
            issue = error.issue
            return failure(issue.code, issue.message, issue.evidence_id)
    intervals, rules = bound_intervals(checks)
    proofs = tuple(sorted({c.artifact_id for c in checks if c.is_proof}))
    hints = tuple(sorted({c.artifact_id for c in checks if not c.is_proof}))
    for interval in intervals:
        if interval.lower is not None and interval.upper is not None and interval.lower > interval.upper:
            return EvidenceBundle(identity, ExecutionStatus.CONFLICT, intervals, tuple(checks), proofs, hints, rules,
                (ClaimIssue(ExecutionStatus.CONFLICT,
                    f'Contradictory checked bounds for {interval.resource.value}; quarantine evidence and audit the trusted checker'),))
    return EvidenceBundle(identity, ExecutionStatus.COMPLETED, intervals, tuple(checks), proofs, hints, rules)
