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


@pytest.mark.parametrize('oversized', ['primal', 'dual'])
def test_checker_bounds_infinite_vectors_before_conversion_and_allocation(monkeypatch, oversized):
    from itertools import repeat
    import nonlocalgames.analysis.certificates as certificates

    def forbidden(*args, **kwargs):
        pytest.fail('invalid vector lengths must fail before conversion or sparse allocation')

    monkeypatch.setattr(certificates, 'rational', forbidden)
    monkeypatch.setattr(certificates, '_nonsignalling_program', forbidden)
    primal = repeat(object()) if oversized == 'primal' else [object()]*16
    dual = repeat(object()) if oversized == 'dual' else [object()]*12
    with pytest.raises(ValueError, match=f'{oversized} dimensions'):
        verify_nonsignalling_certificate(chsh(), primal, dual)


def test_checker_consumes_at_most_expected_length_plus_one(monkeypatch):
    import nonlocalgames.analysis.certificates as certificates
    consumed = []

    def too_long():
        for index in range(17):
            consumed.append(index)
            yield object()
        pytest.fail('checker consumed beyond the bounded dimension probe')

    monkeypatch.setattr(certificates, 'rational', lambda value: pytest.fail('must reject size first'))
    with pytest.raises(ValueError, match='primal dimensions'):
        verify_nonsignalling_certificate(chsh(), too_long(), [0]*12)
    assert consumed == list(range(17))


@pytest.mark.parametrize('budget', [15, 0, -1, True, 1.5])
def test_checker_rejects_invalid_or_insufficient_budget_before_touching_vectors(monkeypatch, budget):
    import nonlocalgames.analysis.certificates as certificates

    class Untouched:
        def __iter__(self):
            pytest.fail('budget failure must not inspect certificate iterables')

    monkeypatch.setattr(certificates, '_nonsignalling_program',
                        lambda *args, **kwargs: pytest.fail('budget failure must precede allocation'))
    with pytest.raises(ValueError, match='limit|positive integer'):
        verify_nonsignalling_certificate(chsh(), Untouched(), Untouched(), max_variables=budget)


@pytest.mark.parametrize('inputs, outputs', [
    ((1,), (1,)), ((3,), (2,)), ((3, 2), (1, 3)),
    ((2, 1, 2), (2, 3, 1)), ((2, 2, 2), (2, 2, 2)),
])
def test_preflight_accepts_valid_certificates_for_heterogeneous_scenarios(inputs, outputs):
    from math import prod
    from nonlocalgames import Game, Scenario
    from nonlocalgames.analysis.nonsignalling import _nonsignalling_program
    game = Game.from_rule(Scenario(inputs, outputs), lambda x, a: 0)
    _, rhs = _nonsignalling_program(game, 100_000)
    primal = (Fraction(1, prod(outputs)),) * game.scenario.entries
    result = verify_nonsignalling_certificate(game, iter(primal), (0 for _ in rhs))
    assert result.value == 0 and result.guarantee == 'exact'
