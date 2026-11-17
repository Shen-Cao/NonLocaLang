from dataclasses import replace
from fractions import Fraction
import json

import cvxpy as cp
import pytest

from nonlocalgames.analysis import classical_value, certify_xor_upper_bound, verify_xor_certificate
from nonlocalgames.certificates.xor_primal import verify_xor_primal_certificate
from nonlocalgames.certificates.xor_primal_discovery import certify_xor_lower_bound
from nonlocalgames.discovery.diagonal_xor3 import diagonal_xor3_game


def test_asymmetric_primal_and_dual_improve_on_symmetric_controls(tmp_path, monkeypatch):
    original = diagonal_xor3_game(Fraction(1, 4))
    sampling = list(original.distribution)
    delta = Fraction(1, 48)
    sampling[1] += delta
    sampling[2] -= delta
    game = replace(original, distribution=sampling, name='Asymmetric three-question XOR')
    options = {'solver': 'SCS', 'eps': 1e-8, 'max_denominator': 1_000_000}
    lower = certify_xor_lower_bound(game, **options)
    upper = certify_xor_upper_bound(game, **options)
    path = tmp_path / 'certificate-pair.json'
    path.write_text(json.dumps({'primal': lower.witness, 'dual': upper.witness}), encoding='utf-8')

    monkeypatch.setattr(cp.Problem, 'solve', lambda *a, **kw: pytest.fail('raw replay called a solver'))
    pair = json.loads(path.read_text(encoding='utf-8'))
    checked_lower = verify_xor_primal_certificate(game, pair['primal']).value
    checked_upper = verify_xor_certificate(game, pair['dual']).value
    assert (lower.value, upper.value) == (checked_lower, checked_upper)
    # The trine has the same winning probability on both modified cells.
    # Reweighting any fixed strategy changes its total score by at most delta.
    trine = Fraction(13, 16)
    assert classical_value(game).value == Fraction(37, 48)
    assert trine < checked_lower <= checked_upper < trine + delta
    assert checked_upper - checked_lower < Fraction(1, 1000)
