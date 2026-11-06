"""Reference realizations for multipartite parity and consistency games."""

from functools import reduce

import numpy as np

from .strategy import QuantumStrategy


def ghz_strategy() -> QuantumStrategy:
    """Three-qubit GHZ state with input 0 measuring X and input 1 measuring Y.

    Outcome bit a labels eigenvalue (-1)^a. On even-parity input tuples the
    output parity is half the input sum, matching ``games.ghz``.
    """
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    vector = np.zeros(8, dtype=complex)
    vector[[0, 7]] = 1 / np.sqrt(2)
    party = tuple(((np.eye(2) + o) / 2, (np.eye(2) - o) / 2) for o in (x, y))
    return QuantumStrategy(np.outer(vector, vector.conj()), (party,) * 3)


def magic_square_strategy() -> QuantumStrategy:
    """Maximally entangled ququarts with commuting row/column measurements.

    The observable grid is [[XI, IX, XX], [IZ, ZI, ZZ], [-XZ, -ZX, YY]].
    Row products are +I and column products are -I. Bob uses transposed
    observables so their intersection outcomes agree on the maximally
    entangled state. Joint eigenspaces are indexed by the even row and odd
    column bit tuples, in exactly the order used by ``games.magic_square``.
    """
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    grid = (
        (np.kron(x, identity), np.kron(identity, x), np.kron(x, x)),
        (np.kron(identity, z), np.kron(z, identity), np.kron(z, z)),
        (-np.kron(x, z), -np.kron(z, x), np.kron(y, y)),
    )
    rows = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
    columns = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1))

    def joint_povm(observables, outcomes):
        return tuple(
            reduce(np.matmul, ((np.eye(4) + (-1)**bit * observable) / 2
                               for bit, observable in zip(bits, observables)))
            for bits in outcomes
        )

    alice = tuple(joint_povm(row, rows) for row in grid)
    bob = tuple(joint_povm(tuple(grid[row][column].T for row in range(3)), columns)
                for column in range(3))
    vector = np.eye(4).reshape(-1) / 2
    return QuantumStrategy(np.outer(vector, vector), (alice, bob))
