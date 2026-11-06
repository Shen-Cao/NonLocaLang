"""Finite-dimensional tensor-product strategies and explicit witnesses.

Validation is numerical. A passing tolerance check and a computed score are
evidence of feasibility, not a certified mathematical bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from math import prod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nonlocalgames import Game, Scenario


def _readonly_matrix(value: object) -> np.ndarray:
    matrix = np.array(value, dtype=complex, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("State and measurement effects must be square matrices.")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Matrices must be nonempty and finite.")
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, eq=False)
class QuantumStrategy:
    """A state and POVMs in party-major, input-major, outcome-major order.

    Local dimensions are inferred from each party's effects. The density matrix
    uses the tensor-factor order of the parties. Arrays are defensively copied
    and read-only; all validation uses absolute ``tolerance``.
    """

    state: np.ndarray
    measurements: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    tolerance: float = 1e-9
    dimensions: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite.")
        state = _readonly_matrix(self.state)
        measurements = tuple(
            tuple(tuple(_readonly_matrix(effect) for effect in povm) for povm in party)
            for party in self.measurements
        )
        if not measurements or any(not party for party in measurements):
            raise ValueError("At least one party and one input per party are required.")
        if any(not povm for party in measurements for povm in party):
            raise ValueError("Every POVM must contain at least one outcome.")
        dimensions = tuple(party[0][0].shape[0] for party in measurements)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "dimensions", dimensions)
        self.validate()

    def validate(self) -> dict[str, float]:
        """Check positivity, Hermiticity, normalization, and tensor dimensions.

        Returns the largest residuals and smallest eigenvalue across the state
        and all effects. Raises ``ValueError`` on failure; no repair is applied.
        """
        if self.state.shape != (prod(self.dimensions),) * 2:
            raise ValueError("State dimension does not match the local tensor factors.")
        hermiticity = 0.0
        minimum_eigenvalue = float("inf")
        normalization = float(abs(np.trace(self.state) - 1))
        matrices = [self.state]
        for dimension, party in zip(self.dimensions, self.measurements):
            for povm in party:
                if any(effect.shape != (dimension, dimension) for effect in povm):
                    raise ValueError("All effects of a party must have the same dimension.")
                normalization = max(
                    normalization,
                    float(np.linalg.norm(sum(povm) - np.eye(dimension), ord=2)),
                )
                matrices.extend(povm)
        for matrix in matrices:
            hermiticity = max(
                hermiticity, float(np.linalg.norm(matrix - matrix.conj().T, ord=2))
            )
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min()),
            )
        if hermiticity > self.tolerance:
            raise ValueError("State or effect is not Hermitian within tolerance.")
        if normalization > self.tolerance:
            raise ValueError("State trace or POVM sum is not normalized within tolerance.")
        if minimum_eigenvalue < -self.tolerance:
            raise ValueError("State or effect is not positive semidefinite within tolerance.")
        return {
            "hermiticity_residual": hermiticity,
            "normalization_residual": normalization,
            "minimum_eigenvalue": minimum_eigenvalue,
        }

    def probabilities(self, scenario: Scenario) -> np.ndarray:
        """Return Born probabilities as a flat input-major/output-minor array.

        Values are not clipped or renormalized; small floating-point residuals
        remain visible. The returned array is numerical evidence only.
        """
        if len(self.measurements) != len(scenario.input_sizes):
            raise ValueError("The strategy and scenario have different party counts.")
        for party, inputs, outputs in zip(
            self.measurements, scenario.input_sizes, scenario.output_sizes
        ):
            if len(party) != inputs or any(len(povm) != outputs for povm in party):
                raise ValueError("The strategy's POVMs do not match the scenario alphabets.")
        probabilities = []
        for x in scenario.inputs:
            for a in scenario.outputs:
                joint = reduce(
                    np.kron,
                    (self.measurements[i][xi][ai] for i, (xi, ai) in enumerate(zip(x, a))),
                )
                value = np.trace(self.state @ joint)
                if abs(value.imag) > self.tolerance:
                    raise ValueError("Born probability has an unexpected imaginary residual.")
                probabilities.append(float(value.real))
        return np.array(probabilities)

    def score(self, game: Game) -> float:
        """Evaluate the expected payoff numerically, without a certificate."""
        coefficients = np.array(
            [float(game.coefficient(x, a)) for x in game.scenario.inputs for a in game.scenario.outputs]
        )
        return float(coefficients @ self.probabilities(game.scenario))

    def conjugate(self) -> QuantumStrategy:
        """Simultaneously conjugate state and POVMs, preserving all probabilities."""
        return QuantumStrategy(
            self.state.conj(),
            tuple(tuple(tuple(e.conj() for e in povm) for povm in p) for p in self.measurements),
            self.tolerance,
        )


def chsh_strategy() -> QuantumStrategy:
    """Bell state with A=(Z,X), B=((Z+X)/sqrt(2),(Z-X)/sqrt(2))."""
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    vector = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    observables = ((z, x), ((z + x) / np.sqrt(2), (z - x) / np.sqrt(2)))
    measurements = tuple(
        tuple(((np.eye(2) + observable) / 2, (np.eye(2) - observable) / 2) for observable in party)
        for party in observables
    )
    return QuantumStrategy(np.outer(vector, vector.conj()), measurements)
