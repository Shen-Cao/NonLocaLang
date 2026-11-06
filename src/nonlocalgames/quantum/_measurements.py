"""Local measurement classes and fixed-dimension initialization/update rules."""
import numpy as np


def validate_measurement_class(measurement_class, output_sizes):
    if measurement_class not in ("povm", "projective"):
        raise ValueError("measurement_class must be 'povm' or 'projective'")
    if measurement_class == "projective" and any(size > 2 for size in output_sizes):
        raise NotImplementedError("Projective search currently supports one or two outputs per party")
    return measurement_class.value if hasattr(measurement_class, "value") else measurement_class


def projective_update(effective):
    """Optimize a one/two-outcome projective measurement by spectral projection.

    For two outcomes the objective is tr(H1)+tr(E0(H0-H1)). The positive
    spectral projector of H0-H1 is an optimum; zero/full ranks are permitted.
    This uses the existing system dimension and introduces no dilation.
    """
    dimension = effective[0].shape[0]
    identity = np.eye(dimension, dtype=complex)
    if len(effective) == 1:
        return (identity,)
    if len(effective) != 2:
        raise NotImplementedError("Projective update needs one or two outcomes")
    difference = effective[0] - effective[1]
    values, vectors = np.linalg.eigh((difference + difference.conj().T) / 2)
    chosen = vectors[:, values > 0]
    first = chosen @ chosen.conj().T
    return first, identity - first


def random_projective(rng, dimension, outputs):
    """Seeded projectors; nontrivial ranks initialize binary systems with d>1.

    Optimization may subsequently choose any rank, including zero and full.
    Single-dimensional binary systems initialize one of the deterministic POVMs.
    """
    identity = np.eye(dimension, dtype=complex)
    if outputs == 1:
        return (identity,)
    if outputs != 2:
        raise NotImplementedError("Projective initialization needs one or two outcomes")
    if dimension == 1:
        first = float(rng.integers(0, 2)) * identity
    else:
        gaussian = rng.normal(size=(dimension, dimension)) + 1j*rng.normal(size=(dimension, dimension))
        vectors, _ = np.linalg.qr(gaussian)
        rank = int(rng.integers(1, dimension))
        first = vectors[:, :rank] @ vectors[:, :rank].conj().T
    return first, identity - first
