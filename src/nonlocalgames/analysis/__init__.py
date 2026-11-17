"""Optimization and independently verified bounds with explicit evidence semantics."""
from .classical import classical_value
from .certificates import certify_nonsignalling, verify_nonsignalling_certificate
from .nonsignalling import nonsignalling_value
from .npa import npa_bound
from .result import AnalysisResult
from .xor import xor_quantum_value
from .xor_template import XorScreenSession
from ..certificates.xor import verify_xor_certificate, verify_xor_dual
from ..certificates.xor_discovery import certify_xor_upper_bound
from ..certificates.xor_primal import verify_xor_gram, verify_xor_primal_certificate
from ..certificates.xor_primal_discovery import certify_xor_lower_bound
from ..certificates.quantum import (certify_quantum_lower_bound, certify_quantum_strategy,
                                    verify_quantum_lower_certificate)

__all__ = ['AnalysisResult', 'classical_value', 'certify_nonsignalling', 'nonsignalling_value',
           'npa_bound', 'verify_nonsignalling_certificate', 'xor_quantum_value',
           'certify_xor_upper_bound', 'verify_xor_dual', 'verify_xor_certificate',
           'certify_quantum_lower_bound', 'certify_quantum_strategy', 'verify_quantum_lower_certificate',
           'verify_xor_gram', 'verify_xor_primal_certificate', 'certify_xor_lower_bound',
           'XorScreenSession']
