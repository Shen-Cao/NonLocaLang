"""Exact coefficient conventions shared by XOR optimization and verification."""
from fractions import Fraction

from ..core.game import Game
from ..core.identity import verifier_digest as game_fingerprint
from .linalg import positive_integer

__all__ = ["xor_coefficients", "cost_matrix", "game_fingerprint"]


def xor_coefficients(game, *, max_matrix_size=128):
    """Return c0,W for score=c0+sum_xy Wxy <u_x,v_y>, using exact rationals."""
    if not isinstance(game, Game):
        raise TypeError("Expected a finite Game")
    scenario = game.scenario
    if scenario.parties != 2 or scenario.output_sizes != (2, 2):
        raise ValueError("XOR SDP requires two parties with two outputs each")
    m, n = scenario.input_sizes
    budget = positive_integer(max_matrix_size, "max_matrix_size")
    if m + n > budget:
        raise ValueError(f"XOR Gram matrix size {m+n} exceeds limit {budget} (max_matrix_size)")
    constant = Fraction(0)
    weights = [[Fraction(0) for _ in range(n)] for _ in range(m)]
    for x in scenario.inputs:
        probability = game.probability(x)
        if probability == 0:
            continue
        even, odd = game.reward(x, (0, 0)), game.reward(x, (0, 1))
        if game.reward(x, (1, 1)) != even or game.reward(x, (1, 0)) != odd:
            raise ValueError("XOR payoff must depend only on answer parity at positive-probability questions")
        constant += probability * (even + odd) / 2
        weights[x[0]][x[1]] = probability * (even - odd) / 2
    return constant, tuple(tuple(row) for row in weights)


def cost_matrix(weights):
    """Symmetric C=[[0,W/2],[W^T/2,0]], so trace(C G)=sum_xy Wxy Gxy."""
    m, n = len(weights), len(weights[0])
    matrix = [[Fraction(0) for _ in range(m+n)] for _ in range(m+n)]
    for x in range(m):
        for y in range(n):
            matrix[x][m+y] = matrix[m+y][x] = weights[x][y] / 2
    return tuple(tuple(row) for row in matrix)
