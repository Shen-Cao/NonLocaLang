"""Claim truth is separate from execution outcome and imported result labels."""
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from numbers import Integral

from ..core.game import rational
from ..core.resources import Resource, ResourceSpec


class TruthStatus(StrEnum):
    PROVED = 'proved'
    DISPROVED = 'disproved'
    UNKNOWN = 'unknown'


class ExecutionStatus(StrEnum):
    COMPLETED = 'completed'
    UNSUPPORTED = 'unsupported'
    BUDGET_EXCEEDED = 'budget_exceeded'
    INVALID_EVIDENCE = 'invalid_evidence'
    CONFLICT = 'conflict'
    CHECKER_ERROR = 'checker_error'
    DEPENDENCY_UNAVAILABLE = 'dependency_unavailable'


class ValueRelation(StrEnum):
    AT_LEAST = 'at_least'
    AT_MOST = 'at_most'


@dataclass(frozen=True)
class ValueClaim:
    resource: ResourceSpec
    relation: ValueRelation
    threshold: Fraction

    def __post_init__(self):
        if not isinstance(self.resource, ResourceSpec):
            raise TypeError('A value claim requires a ResourceSpec')
        if isinstance(self.threshold, bool):
            raise TypeError('Claim thresholds must be exact numbers, not booleans')
        object.__setattr__(self, 'relation', ValueRelation(self.relation))
        object.__setattr__(self, 'threshold', rational(self.threshold))


@dataclass(frozen=True)
class GapClaim:
    higher: ResourceSpec
    lower: ResourceSpec
    threshold: Fraction = Fraction(0)

    def __post_init__(self):
        if not isinstance(self.higher, ResourceSpec) or not isinstance(self.lower, ResourceSpec):
            raise TypeError('A gap claim requires two ResourceSpec objects')
        if isinstance(self.threshold, bool):
            raise TypeError('Claim thresholds must be exact numbers, not booleans')
        threshold = rational(self.threshold)
        if threshold < 0:
            raise ValueError('A strict advantage threshold must be nonnegative')
        object.__setattr__(self, 'threshold', threshold)


@dataclass(frozen=True)
class VerificationBudget:
    """Per-artifact count/size limits; these are not a wall-clock guarantee."""
    max_strategies: int = 1_000_000
    max_variables: int = 100_000
    max_matrix_size: int = 128
    max_artifact_bytes: int = 8_000_000
    max_artifacts: int = 100
    max_property_entries: int = 1_000_000
    max_quantum_joint_dimension: int = 16
    max_quantum_entries: int = 100_000
    max_quantum_events: int = 100_000
    max_quantum_parties: int = 32
    max_quantum_scalar_bits: int = 4096
    max_wire_scalar_bits: int = 4096
    max_wire_scalar_chars: int = 8192

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class CertifiedInterval:
    resource: Resource
    lower: Fraction | None = None
    upper: Fraction | None = None
    lower_refs: tuple[str, ...] = ()
    upper_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimIssue:
    code: ExecutionStatus
    message: str
    evidence_id: str | None = None


@dataclass(frozen=True)
class EvidenceCheck:
    """Checker output only; public inference never accepts it in place of raw evidence."""
    artifact_id: str
    resource: Resource
    lower: Fraction | None
    upper: Fraction | None
    checker: str
    is_proof: bool = True
    attained_score: Fraction | None = None
    actual_dimensions: tuple[int, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, 'resource', Resource(self.resource))
        if not isinstance(self.is_proof, bool):
            raise TypeError('is_proof must be a Boolean')
        for name in ('lower', 'upper', 'attained_score'):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (Fraction, Integral)):
                    raise TypeError('Checked bounds and attainment scores must be exact rationals')
                object.__setattr__(self, name, rational(value))
        if not self.is_proof and any(v is not None for v in
                                    (self.lower, self.upper, self.attained_score, self.actual_dimensions)):
            raise ValueError('Untrusted hints cannot carry checked bound or attainment facts')
        if self.actual_dimensions is not None:
            dimensions = tuple(self.actual_dimensions)
            if self.attained_score is None or not dimensions or any(
                    isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in dimensions):
                raise ValueError('Witness dimensions require an attained score and positive integers')
            object.__setattr__(self, 'actual_dimensions', tuple(int(d) for d in dimensions))


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    game_id: str
    truth: TruthStatus
    execution: ExecutionStatus
    intervals: tuple[CertifiedInterval, ...] = ()
    verified_evidence: tuple[str, ...] = ()
    numerical_hints: tuple[str, ...] = ()
    derivation: tuple[str, ...] = ()
    issues: tuple[ClaimIssue, ...] = ()
    checks: tuple[EvidenceCheck, ...] = ()
    schema: str = field(default='nonlocalgames.claim_assessment.v2', init=False)

    def __post_init__(self):
        object.__setattr__(self, 'truth', TruthStatus(self.truth))
        object.__setattr__(self, 'execution', ExecutionStatus(self.execution))
        if self.execution != ExecutionStatus.COMPLETED and self.truth != TruthStatus.UNKNOWN:
            raise ValueError('An operational failure is not a mathematical verdict')


class EvidenceError(ValueError):
    def __init__(self, code, message, evidence_id=None):
        super().__init__(message)
        self.issue = ClaimIssue(ExecutionStatus(code), message, evidence_id)
