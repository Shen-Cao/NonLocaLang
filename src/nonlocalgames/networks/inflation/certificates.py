"""Solver-independent exact certificates bound to a complete spiral-LP context."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from math import prod
from numbers import Integral

from ...core.game import rational
from ..behavior import positive_budget
from ._program import RESOURCE, ROW_SCHEMA, compile_spiral
from .result import SpiralResult

CERTIFICATE_SCHEMA = "nonlocalgames.classical_spiral_certificate.v1"


@dataclass(frozen=True)
class SpiralCertificate:
    side: str
    coefficients: tuple[Fraction, ...]
    output_sizes: tuple[int, ...]
    network_fingerprint: str
    behavior_fingerprint: str
    row_schema: str = ROW_SCHEMA
    resource: str = RESOURCE


def _validate_metadata(certificate):
    if certificate.row_schema != ROW_SCHEMA or certificate.resource != RESOURCE:
        raise ValueError("Certificate uses a different row schema or resource; classical fanout is required")
    if certificate.side not in ("primal", "dual"):
        raise ValueError("Certificate side must be primal or dual")
    dims = certificate.output_sizes
    if not isinstance(dims, tuple) or len(dims) != 3 or any(
            isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in dims):
        raise ValueError("Certificate output sizes must contain three positive integers")
    for digest in (certificate.network_fingerprint, certificate.behavior_fingerprint):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Certificate context fingerprints must be SHA256 hex strings")


def _vector(values, expected):
    entries = tuple(islice(iter(values), expected + 1))
    if len(entries) != expected:
        raise ValueError("Certificate vector length does not match the marginal program")
    return tuple(map(rational, entries))


def _dual_quantities(program, coefficients):
    matrix = program.matrix.tocsc()
    slacks = tuple(sum((int(matrix.data[k])*coefficients[matrix.indices[k]]
                       for k in range(matrix.indptr[j], matrix.indptr[j+1])), Fraction(0))
                   for j in range(matrix.shape[1]))
    margin = sum((b*y for b, y in zip(program.rhs, coefficients)), Fraction(0))
    return slacks, margin


def _verify_primal(program, coefficients):
    if any(value < 0 for value in coefficients):
        raise ValueError("Primal certificate has a negative probability")
    matrix = program.matrix
    for row, target in enumerate(program.rhs):
        value = sum((int(matrix.data[k])*coefficients[matrix.indices[k]]
                     for k in range(matrix.indptr[row], matrix.indptr[row+1])), Fraction(0))
        if value != target:
            raise ValueError("Primal certificate violates an exact inflated marginal equality")


def make_certificate(program, side, coefficients):
    return SpiralCertificate(side, tuple(coefficients), program.output_sizes,
                             program.network_fingerprint, program.behavior_fingerprint)


def spiral_certificate_to_dict(certificate, *, max_coefficients=100_000):
    if not isinstance(certificate, SpiralCertificate):
        raise TypeError("Expected SpiralCertificate")
    _validate_metadata(certificate)
    limit = positive_budget(max_coefficients, "max_coefficients")
    joint = prod(certificate.output_sizes)
    expected = joint**2 if certificate.side == "primal" else 5*joint
    if expected > limit:
        raise ValueError("Certificate vector exceeds max_coefficients")
    coefficients = _vector(certificate.coefficients, expected)
    return {"schema": CERTIFICATE_SCHEMA, "resource": certificate.resource,
            "row_schema": certificate.row_schema, "side": certificate.side,
            "output_sizes": list(certificate.output_sizes),
            "network_fingerprint": certificate.network_fingerprint,
            "behavior_fingerprint": certificate.behavior_fingerprint,
            "coefficients": [str(value) for value in coefficients]}


def spiral_certificate_from_dict(data, *, max_coefficients=100_000):
    budget = positive_budget(max_coefficients, "max_coefficients")
    fields = {"schema", "resource", "row_schema", "side", "output_sizes", "network_fingerprint",
              "behavior_fingerprint", "coefficients"}
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != CERTIFICATE_SCHEMA:
        raise ValueError("Invalid spiral certificate fields or schema")
    if not isinstance(data["output_sizes"], list):
        raise ValueError("Certificate output sizes must be a JSON list")
    values = data["coefficients"]
    if not isinstance(values, list) or len(values) > budget:
        raise ValueError("Certificate coefficients must be a list within max_coefficients")
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Serialized exact certificate coefficients must be rational strings")
    result = SpiralCertificate(data["side"], tuple(values), tuple(data["output_sizes"]),
                               data["network_fingerprint"], data["behavior_fingerprint"],
                               data["row_schema"], data["resource"])
    _validate_metadata(result)
    joint = prod(result.output_sizes)
    expected = joint**2 if result.side == "primal" else 5*joint
    if len(values) != expected:
        raise ValueError("Certificate vector length does not match its declared output sizes")
    return SpiralCertificate(result.side, tuple(map(rational, values)), result.output_sizes,
                             result.network_fingerprint, result.behavior_fingerprint,
                             result.row_schema, result.resource)


def verify_spiral_certificate(network, behavior, certificate, *, max_variables=100_000,
                               max_equalities=100_000, max_nonzeros=500_000):
    """Verify exact inflated primal feasibility or a classical Farkas contradiction.

    The complete behavior, source order, output alphabets, and matrix ordering
    are bound to the saved certificate. Verification never calls an optimizer.
    A primal witness yields not_ruled_out, not original-network compatibility.
    """
    program = compile_spiral(network, behavior, max_variables=max_variables,
                              max_equalities=max_equalities, max_nonzeros=max_nonzeros)
    if isinstance(certificate, dict):
        certificate = spiral_certificate_from_dict(certificate,
            max_coefficients=max(program.matrix.shape))
    if not isinstance(certificate, SpiralCertificate):
        raise TypeError("Expected SpiralCertificate or its serialized dictionary")
    _validate_metadata(certificate)
    if certificate.output_sizes != program.output_sizes:
        raise ValueError("Certificate output alphabets differ from the supplied behavior")
    if (certificate.network_fingerprint != program.network_fingerprint
            or certificate.behavior_fingerprint != program.behavior_fingerprint):
        raise ValueError("Certificate is bound to a different graph or complete behavior")
    expected = program.matrix.shape[1] if certificate.side == "primal" else program.matrix.shape[0]
    coefficients = _vector(certificate.coefficients, expected)
    checked = make_certificate(program, certificate.side, coefficients)
    diagnostics = {**program.sizes, "certified": True,
                   "scope": "fixed classical fanout spiral relaxation; no quantum inference"}
    if certificate.side == "primal":
        _verify_primal(program, coefficients)
        return SpiralResult("not_ruled_out", "exact_relaxation_feasibility", diagnostics, checked)
    slacks, margin = _dual_quantities(program, coefficients)
    if min(slacks) < 0 or margin >= 0:
        raise ValueError("Dual certificate is not an exact Farkas contradiction")
    diagnostics.update(minimum_dual_slack=min(slacks), farkas_margin=margin)
    return SpiralResult("incompatible", "exact_classical_incompatibility", diagnostics, checked)
