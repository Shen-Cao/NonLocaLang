from fractions import Fraction

import pytest
from scipy.sparse import csr_matrix

from nonlocalgames.analysis import certify_nonsignalling, verify_nonsignalling_certificate
from nonlocalgames.analysis.certificates import _verify_rational_lp
from nonlocalgames.games import chsh, gyni, magic_square


@pytest.mark.parametrize("constructor, expected", [(chsh, Fraction(1)), (gyni, Fraction(1, 3)),
                                                   (magic_square, Fraction(1))])
def test_exact_nonsignalling_primal_dual_certificate(constructor, expected):
    result = certify_nonsignalling(constructor())
    assert result.value == expected
    assert isinstance(result.value, Fraction)
    assert result.guarantee == "exact"
    assert result.diagnostics["certified"]
    assert result.diagnostics["lower_bound"] == result.diagnostics["upper_bound"]
    assert all(isinstance(x, Fraction) for x in result.witness["primal"])


def test_reconstruction_failure_does_not_claim_certificate():
    with pytest.raises(ValueError, match="Reconstructed"):
        certify_nonsignalling(gyni(), max_denominator=1)


def test_exported_certificate_roundtrip_without_solver(monkeypatch):
    import scipy.optimize
    game = gyni()
    witness = certify_nonsignalling(game).witness
    exported = {key: [str(v) for v in values] for key, values in witness.items()}

    def forbidden_solver(*args, **kwargs):
        raise AssertionError("Certificate verification must not solve an optimization")

    monkeypatch.setattr(scipy.optimize, "linprog", forbidden_solver)
    result = verify_nonsignalling_certificate(game, **exported)
    assert result.value == Fraction(1, 3)
    assert result.guarantee == "exact"
    assert result.method == "exact_lp_verification"
    assert result.witness == witness


def test_public_checker_rejects_damaged_or_inexact_witness():
    game = chsh()
    witness = certify_nonsignalling(game).witness
    primal, dual = list(witness["primal"]), list(witness["dual"])
    with pytest.raises(ValueError, match="dimensions"):
        verify_nonsignalling_certificate(game, primal[:-1], dual)
    primal[0] += 1
    with pytest.raises(ValueError, match="equality"):
        verify_nonsignalling_certificate(game, primal, dual)
    with pytest.raises(ValueError, match="inequality"):
        verify_nonsignalling_certificate(game, witness["primal"], [0]*len(dual))
    with pytest.raises(TypeError, match="floating-point"):
        verify_nonsignalling_certificate(game, list(map(float, witness["primal"])), dual)
    with pytest.raises(TypeError, match="floating-point"):
        verify_nonsignalling_certificate(game, witness["primal"], list(map(float, dual)))


def test_public_checker_can_certify_nonzero_interval():
    game = chsh()
    dual = certify_nonsignalling(game).witness["dual"]
    primal = tuple(int(a == (0, 0)) for x in game.inputs for a in game.outputs)
    result = verify_nonsignalling_certificate(game, primal, dual)
    assert result.guarantee == "certified_upper_bound"
    assert result.status == "certified_interval"
    assert result.value == 1
    assert result.diagnostics["lower_bound"] == Fraction(3, 4)
    assert result.diagnostics["upper_bound"] == 1


def test_exact_verifier_rejects_bad_primal_and_dual_and_records_gap():
    matrix = csr_matrix([[1, 1]])
    coefficients = (Fraction(1), Fraction(0))
    assert _verify_rational_lp(matrix, [1], coefficients, (Fraction(0), Fraction(1)),
                               (Fraction(-1),)) == (0, 1)
    with pytest.raises(ValueError, match="negative"):
        _verify_rational_lp(matrix, [1], coefficients, (Fraction(-1), Fraction(2)), (Fraction(-1),))
    with pytest.raises(ValueError, match="equality"):
        _verify_rational_lp(matrix, [1], coefficients, (Fraction(1), Fraction(1)), (Fraction(-1),))
    with pytest.raises(ValueError, match="inequality"):
        _verify_rational_lp(matrix, [1], coefficients, (Fraction(1), Fraction(0)), (Fraction(0),))
