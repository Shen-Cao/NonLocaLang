"""Exact verification paths, separate from numerical discovery."""
from .linalg import PSDVerification, verify_psd
from .xor import verify_xor_certificate, verify_xor_dual
from .xor_discovery import certify_xor_upper_bound

__all__ = ["PSDVerification", "verify_psd", "verify_xor_dual", "verify_xor_certificate",
           "certify_xor_upper_bound"]
