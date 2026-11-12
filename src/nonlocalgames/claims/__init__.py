"""Explicit claims inferred only from independently admitted evidence."""
from .artifacts import (EvidenceArtifact, EvidenceKind, evidence_from_dict,
                        evidence_to_dict, make_evidence)
from .inference import assess_gap, assess_value, claim_id
from .types import (ClaimAssessment, ExecutionStatus, GapClaim, TruthStatus,
                    ValueClaim, ValueRelation, VerificationBudget)
from .verify import verify_artifact

__all__ = ['EvidenceArtifact', 'EvidenceKind', 'evidence_from_dict', 'evidence_to_dict',
           'make_evidence', 'assess_gap', 'assess_value', 'claim_id', 'ClaimAssessment',
           'ExecutionStatus', 'GapClaim', 'TruthStatus', 'ValueClaim', 'ValueRelation',
           'VerificationBudget', 'verify_artifact']

from .serialization import assessment_to_dict, claim_from_dict, claim_to_dict

__all__ += ["assessment_to_dict", "claim_from_dict", "claim_to_dict"]
