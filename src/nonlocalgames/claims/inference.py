"""Conservative claim inference from rechecked artifacts and explicit inclusions."""
from hashlib import sha256
from itertools import islice

from ..core.identity import canonical_json, resource_id, verifier_id
from .resources import STANDARD_CHAIN, includes, is_standard
from .types import (CertifiedInterval, ClaimAssessment, ClaimIssue, EvidenceError,
                    ExecutionStatus, GapClaim, TruthStatus, ValueClaim, ValueRelation,
                    VerificationBudget)
from .verify import verify_artifact


def claim_id(game, claim):
    if not isinstance(claim, (ValueClaim, GapClaim)):
        raise TypeError('Expected ValueClaim or GapClaim')
    payload = {'game_id': verifier_id(game), 'threshold': str(claim.threshold)}
    if isinstance(claim, ValueClaim):
        payload.update(kind='value', resource=resource_id(claim.resource), relation=claim.relation.value)
    elif isinstance(claim, GapClaim):
        payload.update(kind='gap_greater', higher=resource_id(claim.higher), lower=resource_id(claim.lower))
    else:
        raise TypeError('Expected ValueClaim or GapClaim')
    digest = sha256(canonical_json(payload).encode('utf-8')).hexdigest()
    return 'nonlocalgames.claim.v1:' + digest


def _failure(game, claim, code, message, evidence_id=None):
    return ClaimAssessment(claim_id(game, claim), verifier_id(game), TruthStatus.UNKNOWN, code,
                           issues=(ClaimIssue(code, message, evidence_id),))


def _intervals(checks):
    intervals, rules = {}, set()
    for target in STANDARD_CHAIN:
        lowers, uppers = [], []
        for check in checks:
            if not check.is_proof:
                continue
            if check.lower is not None and includes(target, check.resource):
                lowers.append((check.lower, check.artifact_id))
                rules.add(f'lower:{check.resource.value}->{target.value}')
            if check.upper is not None and includes(check.resource, target):
                uppers.append((check.upper, check.artifact_id))
                rules.add(f'upper:{check.resource.value}->{target.value}')
        lower = max((v for v, _ in lowers), default=None)
        upper = min((v for v, _ in uppers), default=None)
        intervals[target] = CertifiedInterval(target, lower, upper,
            tuple(sorted({ref for v, ref in lowers if v == lower})),
            tuple(sorted({ref for v, ref in uppers if v == upper})))
    return intervals, tuple(sorted(rules))


def _prepare(game, claim, artifacts, budget):
    budget = VerificationBudget() if budget is None else budget
    if not isinstance(budget, VerificationBudget):
        raise TypeError('Expected VerificationBudget')
    resources = (claim.resource,) if isinstance(claim, ValueClaim) else (claim.higher, claim.lower)
    if any(not is_standard(r) for r in resources):
        return _failure(game, claim, ExecutionStatus.UNSUPPORTED,
                        'Claim inference currently covers only the standard unrestricted resource chain')
    if isinstance(claim, GapClaim) and not includes(claim.higher.kind, claim.lower.kind):
        return _failure(game, claim, ExecutionStatus.UNSUPPORTED,
                        'This gap claim does not follow the supported resource inclusion order')
    try:
        artifacts = tuple(islice(iter(artifacts), budget.max_artifacts + 1))
    except TypeError:
        return _failure(game, claim, ExecutionStatus.INVALID_EVIDENCE, 'Evidence must be an iterable of data artifacts')
    if len(artifacts) > budget.max_artifacts:
        return _failure(game, claim, ExecutionStatus.BUDGET_EXCEEDED, 'Evidence list exceeds max_artifacts')
    checks = []
    for artifact in artifacts:
        try:
            checks.append(verify_artifact(game, artifact, budget=budget))
        except EvidenceError as error:
            issue = error.issue
            return _failure(game, claim, issue.code, issue.message, issue.evidence_id)
    intervals, rules = _intervals(checks)
    proofs = tuple(sorted({c.artifact_id for c in checks if c.is_proof}))
    hints = tuple(sorted({c.artifact_id for c in checks if not c.is_proof}))
    for interval in intervals.values():
        if interval.lower is not None and interval.upper is not None and interval.lower > interval.upper:
            return ClaimAssessment(claim_id(game, claim), verifier_id(game), TruthStatus.UNKNOWN,
                ExecutionStatus.CONFLICT, tuple(intervals.values()), proofs, hints, rules,
                (ClaimIssue(ExecutionStatus.CONFLICT,
                    f'Contradictory checked bounds for {interval.resource.value}; quarantine evidence and audit the trusted checker'),), tuple(checks))
    return intervals, rules, proofs, hints, tuple(checks)


def assess_value(game, claim, artifacts=(), *, budget=None):
    """Resolve a threshold claim when certified bounds suffice; hints never suffice."""
    if not isinstance(claim, ValueClaim):
        raise TypeError('Expected ValueClaim')
    prepared = _prepare(game, claim, artifacts, budget)
    if isinstance(prepared, ClaimAssessment):
        return prepared
    intervals, rules, proofs, hints, checks = prepared
    interval = intervals[claim.resource.kind]
    lower, upper, threshold = interval.lower, interval.upper, claim.threshold
    truth = TruthStatus.UNKNOWN
    if claim.relation == ValueRelation.AT_LEAST:
        if lower is not None and lower >= threshold:
            truth = TruthStatus.PROVED
        elif upper is not None and upper < threshold:
            truth = TruthStatus.DISPROVED
    else:
        if upper is not None and upper <= threshold:
            truth = TruthStatus.PROVED
        elif lower is not None and lower > threshold:
            truth = TruthStatus.DISPROVED
    return ClaimAssessment(claim_id(game, claim), verifier_id(game), truth,
        ExecutionStatus.COMPLETED, (interval,), proofs, hints, rules, checks=checks)


def assess_gap(game, claim, artifacts=(), *, budget=None):
    """For a strict gap, prove L_high-U_low>epsilon or disprove U_high-L_low<=epsilon."""
    if not isinstance(claim, GapClaim):
        raise TypeError('Expected GapClaim')
    prepared = _prepare(game, claim, artifacts, budget)
    if isinstance(prepared, ClaimAssessment):
        return prepared
    intervals, rules, proofs, hints, checks = prepared
    high, low = intervals[claim.higher.kind], intervals[claim.lower.kind]
    truth = TruthStatus.UNKNOWN
    if claim.higher.kind == claim.lower.kind:
        truth, rules = TruthStatus.DISPROVED, rules + ('identical_resource_gap_zero',)
    elif high.lower is not None and low.upper is not None and high.lower - low.upper > claim.threshold:
        truth = TruthStatus.PROVED
    elif high.upper is not None and low.lower is not None and high.upper - low.lower <= claim.threshold:
        truth = TruthStatus.DISPROVED
    return ClaimAssessment(claim_id(game, claim), verifier_id(game), truth,
        ExecutionStatus.COMPLETED, (high, low), proofs, hints, rules, checks=checks)
