"""Precise property definitions and immutable assessment receipts."""
from dataclasses import dataclass, field
from enum import StrEnum

from ..claims.types import (CertifiedInterval, ClaimIssue, EvidenceCheck, ExecutionStatus,
                            TruthStatus)
from ..core.resources import ResourceSpec


class PropertyKind(StrEnum):
    ACCEPTANCE = 'acceptance'
    BOOLEAN_PREDICATE = 'boolean_predicate'
    XOR_PREDICATE = 'xor_predicate'
    XOR_SCORE = 'xor_score'
    PERFECT_STRATEGY = 'perfect_strategy'
    PSEUDOTELEPATHY = 'pseudotelepathy'


class OffSupportPolicy(StrEnum):
    ALL = 'all'
    SAMPLED = 'sampled'


STRUCTURAL_KINDS = frozenset((PropertyKind.ACCEPTANCE, PropertyKind.BOOLEAN_PREDICATE,
                              PropertyKind.XOR_PREDICATE, PropertyKind.XOR_SCORE))


@dataclass(frozen=True)
class PropertyClaim:
    """Structural questions specify support scope; perfection concerns sampled play.

    Pseudotelepathy means an attained perfect finite-dimensional quantum strategy
    and no perfect ordinary local strategy. It has fixed standard resource models.
    """
    kind: PropertyKind
    resource: ResourceSpec | None = None
    off_support: OffSupportPolicy | None = field(default=None, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, 'kind', PropertyKind(self.kind))
        if self.kind in STRUCTURAL_KINDS:
            if self.resource is not None:
                raise ValueError('Structural properties describe the verifier, not a resource model')
            policy = OffSupportPolicy.ALL if self.off_support is None else OffSupportPolicy(self.off_support)
            object.__setattr__(self, 'off_support', policy)
        else:
            if self.off_support is not None:
                raise ValueError('Off-support policy is only a structural property option; perfection concerns sampled play')
            if self.kind == PropertyKind.PERFECT_STRATEGY:
                if not isinstance(self.resource, ResourceSpec):
                    raise TypeError('Perfect-strategy claims require an explicit ResourceSpec')
            elif self.resource is not None:
                raise ValueError('Pseudotelepathy uses the standard local and finite quantum models')


@dataclass(frozen=True)
class StructuralCounterexample:
    """A location/reason to replay against the bound verifier; no unchecked proof token."""
    reason: str
    input_profile: tuple[int, ...] | None = None
    output_profile: tuple[int, ...] | None = None
    party: int | None = None

    def __post_init__(self):
        for name in ('input_profile', 'output_profile'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class PropertyAssessment:
    property_id: str
    game_id: str
    truth: TruthStatus
    execution: ExecutionStatus
    checked_entries: int = 0
    counterexample: StructuralCounterexample | None = None
    intervals: tuple[CertifiedInterval, ...] = ()
    verified_evidence: tuple[str, ...] = ()
    numerical_hints: tuple[str, ...] = ()
    derivation: tuple[str, ...] = ()
    issues: tuple[ClaimIssue, ...] = ()
    checks: tuple[EvidenceCheck, ...] = ()
    schema: str = field(default='nonlocalgames.property_assessment.v1', init=False)

    def __post_init__(self):
        object.__setattr__(self, 'truth', TruthStatus(self.truth))
        object.__setattr__(self, 'execution', ExecutionStatus(self.execution))
        if self.execution != ExecutionStatus.COMPLETED and self.truth != TruthStatus.UNKNOWN:
            raise ValueError('An operational failure is not a mathematical property verdict')
        for name in ('intervals', 'verified_evidence', 'numerical_hints', 'derivation', 'issues', 'checks'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
