"""Finite state specifications and tensor operations with explicit factor order."""
from dataclasses import dataclass
from functools import reduce
from math import prod
from numbers import Integral

import numpy as np

from ..core.numeric_data import positive_tolerance


def validate_dimensions(dimensions):
    dimensions = tuple(dimensions)
    if not dimensions or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1
                             for d in dimensions):
        raise ValueError("dimensions must contain positive integers")
    return tuple(int(d) for d in dimensions)


def _indices(indices, count):
    indices = tuple(indices)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < count
           for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Subsystem indices must be distinct integers within the tensor factors")
    return tuple(int(i) for i in indices)


def validate_ppt_cuts(cuts, parties):
    """Canonicalize proper nonempty cuts; complementary cuts impose the same PPT condition."""
    result = []
    for cut in cuts:
        cut = tuple(sorted(_indices(cut, parties)))
        if not cut or len(cut) == parties:
            raise ValueError("PPT cuts must be nonempty proper subsets of parties")
        complement = tuple(i for i in range(parties) if i not in cut)
        canonical = min(cut, complement)
        if canonical in result:
            raise ValueError("Duplicate or complementary PPT cuts are redundant")
        result.append(canonical)
    return tuple(sorted(result))


def _matrix(state, dimensions):
    matrix = np.asarray(state, dtype=complex)
    size = prod(dimensions)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError("State matrix must be finite and match its tensor dimensions")
    return matrix


def partial_trace(state, dimensions, keep):
    """Trace out other factors, returning the kept factors in the requested order.

    Keeping no factors returns a 1x1 matrix containing the trace.
    """
    dimensions = validate_dimensions(dimensions)
    keep = _indices(keep, len(dimensions))
    traced = tuple(i for i in range(len(dimensions)) if i not in keep)
    order = keep + traced
    tensor = _matrix(state, dimensions).reshape(dimensions + dimensions)
    tensor = tensor.transpose(order + tuple(i + len(dimensions) for i in order))
    kept_size = prod(dimensions[i] for i in keep)
    traced_size = prod(dimensions[i] for i in traced)
    return np.trace(tensor.reshape(kept_size, traced_size, kept_size, traced_size),
                    axis1=1, axis2=3)


def partial_transpose(state, dimensions, parties):
    """Transpose selected local factors in their computational bases."""
    dimensions = validate_dimensions(dimensions)
    parties = _indices(parties, len(dimensions))
    order = list(range(2 * len(dimensions)))
    for i in parties:
        order[i], order[i + len(dimensions)] = order[i + len(dimensions)], order[i]
    size = prod(dimensions)
    return _matrix(state, dimensions).reshape(dimensions + dimensions).transpose(order).reshape(size, size)


def state_diagnostics(state, dimensions, *, tolerance=1e-9, ppt_cuts=()):
    """Validate a density matrix and optional PPT cuts numerically, without repair."""
    dimensions = validate_dimensions(dimensions)
    tolerance = positive_tolerance(tolerance)
    matrix = _matrix(state, dimensions)
    hermiticity = float(np.linalg.norm(matrix - matrix.conj().T, ord=2))
    trace = float(abs(np.trace(matrix) - 1))
    minimum = float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min())
    if hermiticity > tolerance or trace > tolerance or minimum < -tolerance:
        raise ValueError("State violates Hermiticity, trace, or positivity within tolerance")
    diagnostics = {"hermiticity_residual": hermiticity, "trace_residual": trace,
                   "minimum_eigenvalue": minimum}
    cuts = validate_ppt_cuts(ppt_cuts, len(dimensions))
    if cuts:
        ppt_minimum = min(float(np.linalg.eigvalsh(partial_transpose(
            (matrix + matrix.conj().T) / 2, dimensions, cut)).min()) for cut in cuts)
        diagnostics["minimum_ppt_eigenvalue"] = ppt_minimum
        if ppt_minimum < -tolerance:
            raise ValueError("State violates a declared PPT constraint within tolerance")
    return diagnostics


def product_state_residual(state, dimensions):
    """Operator-norm distance from the product of all single-party marginals."""
    dimensions = validate_dimensions(dimensions)
    matrix = _matrix(state, dimensions)
    marginals = [partial_trace(matrix, dimensions, (i,)) for i in range(len(dimensions))]
    return float(np.linalg.norm(matrix - reduce(np.kron, marginals), ord=2))


@dataclass(frozen=True, eq=False)
class FixedState:
    """A numerically validated supplied state with a fixed local tensor factorization.

    The bytes-backed matrix is immutable. Neither this specification nor its
    validation tolerance is a rigorous feasibility certificate.
    """
    state: np.ndarray
    dimensions: tuple[int, ...]
    tolerance: float = 1e-9

    def __post_init__(self):
        dimensions = validate_dimensions(self.dimensions)
        matrix = np.array(self.state, dtype=complex, copy=True)
        state_diagnostics(matrix, dimensions, tolerance=self.tolerance)
        matrix = np.frombuffer(matrix.tobytes(), dtype=complex).reshape(matrix.shape)
        object.__setattr__(self, "state", matrix)
        object.__setattr__(self, "dimensions", dimensions)

    def validate(self, *, ppt_cuts=()):
        return state_diagnostics(self.state, self.dimensions, tolerance=self.tolerance,
                                 ppt_cuts=ppt_cuts)
