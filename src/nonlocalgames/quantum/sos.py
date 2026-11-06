"""Exact CHSH identities in the algebra of commuting-party involutions.

The reduction rules are A_x^2=B_y^2=I and [A_x,B_y]=0. No same-party
commutation or anticommutation is assumed. SymPy is imported only on demand.
This is a certificate for the canonical CHSH expression, not an arbitrary game.
"""

from __future__ import annotations

from dataclasses import dataclass


def _reduce_word(word: tuple[str, ...]) -> tuple[str, ...]:
    """Commute parties past each other and cancel adjacent involutions."""
    result = []
    for party in ("A", "B"):
        stack: list[str] = []
        for letter in word:
            if letter.startswith(party):
                if stack and stack[-1] == letter:
                    stack.pop()
                else:
                    stack.append(letter)
        result.extend(stack)
    return tuple(result)


@dataclass(frozen=True)
class CHSHCertificate:
    """Exact algebraic checks, including an attaining reference realization."""

    verified: bool
    bell_bound: str
    winning_probability: str
    residuals: dict[str, str]
    assumptions: tuple[str, ...]


def verify_chsh_sos() -> CHSHCertificate:
    r"""Verify CHSH's SOS, its square, and exact Bell-state attainment.

    With S=A0(B0+B1)+A1(B0-B1), the identity checked is

        2 sqrt(2) I - S = ((A0-(B0+B1)/sqrt(2))^2
                            + (A1-(B0-B1)/sqrt(2))^2) / sqrt(2).

    Hermiticity makes the right side positive. At saturation both square
    factors annihilate the state. The verified anticommutators of
    (B0+B1)/sqrt(2) and (B0-B1)/sqrt(2) then yield the usual state-dependent
    rigidity relations; this function alone is not a robust self-testing proof.
    """
    import sympy as sp

    def clean(polynomial: dict) -> dict:
        return {word: value for word, coefficient in polynomial.items() if (value := sp.simplify(coefficient)) != 0}

    def add(*polynomials: dict) -> dict:
        result: dict = {}
        for polynomial in polynomials:
            for word, coefficient in polynomial.items():
                result[word] = result.get(word, 0) + coefficient
        return clean(result)

    def scale(coefficient: object, polynomial: dict) -> dict:
        return clean({word: coefficient * value for word, value in polynomial.items()})

    def multiply(left: dict, right: dict) -> dict:
        result: dict = {}
        for lw, lc in left.items():
            for rw, rc in right.items():
                word = _reduce_word(lw + rw)
                result[word] = result.get(word, 0) + lc * rc
        return clean(result)

    def square(polynomial: dict) -> dict:
        return multiply(polynomial, polynomial)

    def commutator(left: dict, right: dict) -> dict:
        return add(multiply(left, right), scale(-1, multiply(right, left)))

    def anticommutator(left: dict, right: dict) -> dict:
        return add(multiply(left, right), multiply(right, left))

    identity = {(): sp.Integer(1)}
    a0, a1, b0, b1 = ({(name,): sp.Integer(1)} for name in ("A0", "A1", "B0", "B1"))
    sqrt2 = sp.sqrt(2)
    bplus = scale(1 / sqrt2, add(b0, b1))
    bminus = scale(1 / sqrt2, add(b0, scale(-1, b1)))
    s = scale(sqrt2, add(multiply(a0, bplus), multiply(a1, bminus)))
    d0 = add(a0, scale(-1, bplus))
    d1 = add(a1, scale(-1, bminus))
    sos = scale(1 / sqrt2, add(square(d0), square(d1)))
    algebra_residuals = {
        "sum_of_squares": add(scale(2 * sqrt2, identity), scale(-1, s), scale(-1, sos)),
        "chsh_square": add(square(s), scale(-4, identity), multiply(commutator(a0, a1), commutator(b0, b1))),
        "bob_combination_anticommutator": anticommutator(bplus, bminus),
    }
    x = sp.Matrix([[0, 1], [1, 0]])
    z = sp.diag(1, -1)
    bob0, bob1 = (z + x) / sqrt2, (z - x) / sqrt2
    reference_s = sp.kronecker_product(z, bob0 + bob1) + sp.kronecker_product(x, bob0 - bob1)
    phi = sp.Matrix([1, 0, 0, 1]) / sqrt2
    attainment = sp.simplify((phi.T * reference_s * phi)[0] - 2 * sqrt2)
    residuals = {name: "0" if not value else str(value) for name, value in algebra_residuals.items()}
    residuals["reference_attainment"] = str(attainment)
    return CHSHCertificate(
        verified=all(not residual for residual in algebra_residuals.values()) and attainment == 0,
        bell_bound=str(2 * sqrt2),
        winning_probability=str(sp.Rational(1, 2) + sqrt2 / 4),
        residuals=residuals,
        assumptions=(
            "A0, A1, B0, B1 are Hermitian involutions",
            "Alice's operators commute with Bob's operators",
            "CHSH inputs are uniform and winning means a xor b equals x*y",
        ),
    )
