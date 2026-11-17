"""Exact verification paths, separate from numerical discovery."""
from .linalg import PSDVerification, verify_psd
from .xor import verify_xor_certificate, verify_xor_dual
from .xor_discovery import certify_xor_upper_bound
from .xor_primal import verify_xor_gram, verify_xor_primal_certificate
from .xor_primal_discovery import certify_xor_lower_bound

__all__ = ["PSDVerification", "verify_psd", "verify_xor_dual", "verify_xor_certificate",
           "certify_xor_upper_bound", "verify_xor_gram", "verify_xor_primal_certificate",
           "certify_xor_lower_bound"]

from .quantum import (certify_quantum_lower_bound, certify_quantum_strategy,
                      verify_quantum_lower_certificate)

__all__ += ['certify_quantum_lower_bound', 'certify_quantum_strategy', 'verify_quantum_lower_certificate']
