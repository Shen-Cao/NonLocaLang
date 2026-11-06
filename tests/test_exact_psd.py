"""Exact PSD verification exercises singular matrices and rejects tiny negative directions."""
from fractions import Fraction

import pytest

from nonlocalgames.certificates import verify_psd


@pytest.mark.parametrize('matrix, rank', [
    ([], 0), ([[0]], 0), ([[2]], 1),
    ([[0, 0], [0, 1]], 1),
    ([[1, 1], [1, 1]], 1),
    ([[Fraction(1, 9), Fraction(1, 3)], [Fraction(1, 3), 1]], 1),
    ([[2, 1], [1, 2]], 2),
])
def test_exact_psd_rank_and_zero_pivots(matrix, rank):
    original = [list(row) for row in matrix]
    checked = verify_psd(matrix)
    assert checked.rank == rank
    assert all(isinstance(p, Fraction) and p >= 0 for p in checked.pivots)
    assert matrix == original


@pytest.mark.parametrize('matrix', [
    [[-Fraction(1, 10**100)]], [[0, 1], [1, 0]],
    [[1, 1], [1, 1-Fraction(1, 10**100)]],
    [[1, 0], [0, -1]],
])
def test_exact_psd_rejects_indefinite_or_negative_matrices(matrix):
    with pytest.raises(ValueError, match='not PSD'):
        verify_psd(matrix)


def test_exact_psd_checks_symmetry_shape_type_and_budget():
    with pytest.raises(ValueError, match='symmetry'):
        verify_psd([[1, 0], [1, 1]])
    with pytest.raises(ValueError, match='square'):
        verify_psd([[1, 0]])
    with pytest.raises(TypeError):
        verify_psd([[1.0]])
    with pytest.raises(ValueError, match='max_size'):
        verify_psd([[1, 0], [0, 1]], max_size=1)
    with pytest.raises(ValueError, match='positive integer'):
        verify_psd([[1]], max_size=True)
