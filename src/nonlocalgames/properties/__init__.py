"""Exact structural recognition and explicitly witnessed nonlocal-game properties."""
from .identity import property_id
from .inference import assess_property
from .serialization import (property_assessment_to_dict, property_claim_from_dict,
                            property_claim_to_dict)
from .types import (OffSupportPolicy, PropertyAssessment, PropertyClaim, PropertyKind,
                    StructuralCounterexample)

__all__ = ['PropertyClaim', 'PropertyKind', 'OffSupportPolicy', 'PropertyAssessment',
           'StructuralCounterexample', 'assess_property', 'property_id',
           'property_claim_to_dict', 'property_claim_from_dict', 'property_assessment_to_dict']
