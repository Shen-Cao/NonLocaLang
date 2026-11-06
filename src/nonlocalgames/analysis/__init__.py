"""Exact local optimization and explicitly numerical convex relaxations."""

from .classical import classical_value
from .certificates import certify_nonsignalling, verify_nonsignalling_certificate
from .nonsignalling import nonsignalling_value
from .npa import npa_bound
from .result import AnalysisResult
from .xor import xor_quantum_value

__all__ = ["AnalysisResult", "classical_value", "certify_nonsignalling", "nonsignalling_value",
           "npa_bound", "verify_nonsignalling_certificate", "xor_quantum_value"]

from ..certificates import certify_xor_upper_bound, verify_xor_certificate, verify_xor_dual

__all__ += ["certify_xor_upper_bound", "verify_xor_dual", "verify_xor_certificate"]
