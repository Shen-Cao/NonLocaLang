"""Narrow property rules built on shared, fail-closed evidence admission."""
from ..claims.resources import is_standard
from ..claims.types import ClaimIssue, ExecutionStatus, TruthStatus, VerificationBudget
from ..core.game import Game
from ..core.identity import verifier_id
from ..core.resources import Resource
from .identity import property_id
from .structural import scan_structure
from .types import (OffSupportPolicy, PropertyAssessment, PropertyClaim, PropertyKind,
                    STRUCTURAL_KINDS)


def _result(game, claim, truth, execution=ExecutionStatus.COMPLETED, *, bundle=None, **fields):
    if bundle is not None:
        fields = {**{name: getattr(bundle, name) for name in (
            'intervals', 'verified_evidence', 'numerical_hints', 'derivation', 'issues', 'checks')}, **fields}
    return PropertyAssessment(property_id(game, claim), verifier_id(game), truth, execution, **fields)


def _failure(game, claim, execution, message, **fields):
    return _result(game, claim, TruthStatus.UNKNOWN, execution,
                   issues=(ClaimIssue(execution, message),), **fields)


def _attaining(checks, resources):
    return tuple(c.artifact_id for c in checks if c.is_proof and c.resource in resources
                 and c.attained_score == 1)


def _perfect(resource, bundle):
    sources = (Resource.LOCAL,) if resource == Resource.LOCAL else (Resource.LOCAL, Resource.QUANTUM_FINITE)
    witnesses = _attaining(bundle.checks, sources)
    if witnesses:
        return TruthStatus.PROVED, ('checked_attainment_of_one:' + resource.value,)
    interval = next(i for i in bundle.intervals if i.resource == resource)
    if interval.upper is not None and interval.upper < 1:
        return TruthStatus.DISPROVED, ('checked_upper_bound_strictly_below_one:' + resource.value,)
    return TruthStatus.UNKNOWN, ('no_checked_attaining_witness_or_excluding_upper_bound',)


def _acceptance_conflict(bundle):
    """An acceptance verifier cannot realize a score or value outside [0,1]."""
    for check in bundle.checks:
        if not check.is_proof:
            continue
        impossible = ((check.lower is not None and check.lower > 1)
                      or (check.upper is not None and check.upper < 0)
                      or (check.attained_score is not None and not 0 <= check.attained_score <= 1))
        if impossible:
            return ClaimIssue(ExecutionStatus.CONFLICT,
                'Checked evidence contradicts the sampled acceptance range [0,1]; quarantine the checker output',
                check.artifact_id)
    for interval in bundle.intervals:
        if interval.lower is not None and interval.lower > 1:
            refs = interval.lower_refs
        elif interval.upper is not None and interval.upper < 0:
            refs = interval.upper_refs
        else:
            continue
        return ClaimIssue(ExecutionStatus.CONFLICT,
            'Checked bounds contradict the sampled acceptance range [0,1]; quarantine the checker output',
            refs[0] if refs else None)
    return None


def assess_property(game, claim, artifacts=(), *, budget=None):
    """Assess exact syntax or rechecked witness properties without new strategy search.

    Local enumeration evidence can trigger budgeted exhaustive replay.
    Structural claims require an empty artifact collection. Perfect-strategy and
    pseudotelepathy claims reverify raw artifacts through the M0 admission path.
    Finite quantum attainment never follows merely from a value lower bound one.
    """
    if not isinstance(game, Game) or not isinstance(claim, PropertyClaim):
        raise TypeError('Expected Game and PropertyClaim')
    budget = VerificationBudget() if budget is None else budget
    if not isinstance(budget, VerificationBudget):
        raise TypeError('Expected VerificationBudget')
    if game.scenario.entries > budget.max_property_entries:
        return _failure(game, claim, ExecutionStatus.BUDGET_EXCEEDED,
                        'Dense verifier table exceeds max_property_entries')
    if claim.kind in STRUCTURAL_KINDS:
        try:
            iterator = iter(artifacts)
            sentinel = object()
            first = next(iterator, sentinel)
        except TypeError:
            return _failure(game, claim, ExecutionStatus.INVALID_EVIDENCE,
                            'Structural artifact collection must be an empty iterable')
        if first is not sentinel:
            return _failure(game, claim, ExecutionStatus.UNSUPPORTED,
                            'Structural claims require no external artifacts; assess evidence separately')
        checked = scan_structure(game, claim.kind, claim.off_support)
        return _result(game, claim, checked.truth, checked_entries=checked.entries,
                       counterexample=checked.counterexample, derivation=(checked.rule,))
    if claim.kind == PropertyKind.PERFECT_STRATEGY and (
        claim.resource.kind not in (Resource.LOCAL, Resource.QUANTUM_FINITE)
        or not is_standard(claim.resource)
    ):
        return _failure(game, claim, ExecutionStatus.UNSUPPORTED,
                        'Attainment claims currently support only standard local or finite quantum resources')
    acceptance = scan_structure(game, PropertyKind.ACCEPTANCE, OffSupportPolicy.SAMPLED)
    if acceptance.truth != TruthStatus.PROVED:
        return _failure(game, claim, ExecutionStatus.UNSUPPORTED,
                        'Perfect play requires acceptance rewards in [0,1] on sampled inputs',
                        checked_entries=acceptance.entries, counterexample=acceptance.counterexample)
    from ..claims.admission import verify_evidence_bundle
    bundle = verify_evidence_bundle(game, artifacts, budget=budget)
    if bundle.execution != ExecutionStatus.COMPLETED:
        return _result(game, claim, TruthStatus.UNKNOWN, bundle.execution, bundle=bundle,
                       checked_entries=acceptance.entries)
    conflict = _acceptance_conflict(bundle)
    if conflict is not None:
        return _result(game, claim, TruthStatus.UNKNOWN, ExecutionStatus.CONFLICT, bundle=bundle,
                       checked_entries=acceptance.entries, issues=bundle.issues + (conflict,))
    if claim.kind == PropertyKind.PERFECT_STRATEGY:
        truth, rules = _perfect(claim.resource.kind, bundle)
    else:
        quantum, qrules = _perfect(Resource.QUANTUM_FINITE, bundle)
        local = next(i for i in bundle.intervals if i.resource == Resource.LOCAL)
        if _attaining(bundle.checks, (Resource.LOCAL,)):
            truth, rules = TruthStatus.DISPROVED, ('checked_perfect_local_strategy_excludes_pseudotelepathy',)
        elif quantum == TruthStatus.DISPROVED:
            truth, rules = TruthStatus.DISPROVED, qrules + ('no_perfect_finite_quantum_strategy',)
        elif quantum == TruthStatus.PROVED and local.upper is not None and local.upper < 1:
            truth, rules = TruthStatus.PROVED, qrules + ('checked_local_upper_bound_strictly_below_one',)
        else:
            truth, rules = TruthStatus.UNKNOWN, ('pseudotelepathy_needs_finite_attainment_and_classical_exclusion',)
    return _result(game, claim, truth, bundle=bundle, checked_entries=acceptance.entries,
                   derivation=bundle.derivation + rules)
