"""Exact supplied-strategy point evaluation; the private verifier is not a player."""
from fractions import Fraction

from ...quantum.exact import verify_psd_complex
from ...quantum.exact.algebra import (
    add, adjoint, identity, kron, multiply, scale, subtract, trace_product,
)
from ...quantum.exact.scalars import exact_scalar
from ._plan import prepare
from .model import LocalHistory, LocalStrategy, ReplayLimits, ReplayResult, Transcript, TwoRoundProtocol


def _check_physics(cases, first, final, ir, iq1, ilocal, b):
    for _, case in cases:
        if case.state.trace(budget=b) != (1, 0):
            raise ValueError("Verifier's joint preparation must have trace one")
        verify_psd_complex(case.state, budget=b)
        for cont in case.after_first:
            for effect in cont.accept:
                verify_psd_complex(effect, budget=b)
                verify_psd_complex(subtract(ir, effect, budget=b), budget=b)

    zero = scale(iq1, 0, budget=b)
    for outcomes in first.values():
        total = zero
        for terms in outcomes:
            for w, k in terms:
                gram = multiply(adjoint(k, budget=b), k, budget=b)
                total = add(total, scale(gram, w, budget=b), budget=b)
        if total != iq1:
            raise ValueError("First instrument must be trace preserving when summed over outcomes")
    if ilocal is not iq1:
        zero = scale(ilocal, 0, budget=b)
    for effects in final.values():
        total = zero
        for effect in effects:
            verify_psd_complex(effect, budget=b)
            total = add(total, effect, budget=b)
        if total != ilocal:
            raise ValueError("Final POVM effects must sum to identity")


def _branch(state, terms, i_untouched, zero, b):
    # The instrument acts only on Q1. Keep the subnormalized R_keep ⊗ Q2 ⊗ M
    # state, even on impossible outcomes. Transferring Q2 changes ownership,
    # not the state or its tensor order.
    result = zero
    for w, k in terms:
        lifted = kron(i_untouched, k, budget=b)
        evolved = multiply(multiply(lifted, state, budget=b), adjoint(lifted, budget=b), budget=b)
        result = add(result, scale(evolved, w, budget=b), budget=b)
    return result


def _mass(prior, state, verifier_effect, local_effect, b):
    real, imag = trace_product(kron(verifier_effect, local_effect, budget=b), state, budget=b)
    if imag:
        raise ArithmeticError("Non-real probability after exact validation")
    return exact_scalar(prior * real, max_bits=b.max_scalar_bits)


def replay(protocol: TwoRoundProtocol, strategy: LocalStrategy, *,
           limits: ReplayLimits = ReplayLimits()) -> ReplayResult:
    """Check all declared data, then enumerate all leaves including zero-mass leaves.

    No strategy callback runs. A delivered classical question selects the first
    instrument; only LocalHistory selects the final POVM. The unnormalized maps
    act only on Q1, preserving correlations with the later wire Q2 and the
    verifier reference. The final POVM acts on Q2 ⊗ M, never on the reference.
    """
    (r, q2, q1), cases, first, final = prepare(protocol, strategy, limits)
    b = limits.exact
    ir, iq1 = identity(r, budget=b), identity(q1, budget=b)
    ilocal = iq1 if q2 == 1 else identity(q2*q1, budget=b)
    _check_physics(cases, first, final, ir, iq1, ilocal, b)
    i_untouched = ir if q2 == 1 else identity(r*q2, budget=b)
    zero = scale(identity(r*q2*q1, budget=b), 0, budget=b)
    transcripts = []
    total = score = Fraction(0)
    for index, (prior, case) in enumerate(cases):
        for y, terms in enumerate(first[case.question]):
            state = _branch(case.state, terms, i_untouched, zero, b)
            cont = case.after_first[y]  # private verifier acts only after y
            h = LocalHistory(case.question, y, cont.question)
            for z, (povm, accept) in enumerate(zip(final[h], cont.accept, strict=True)):
                p = _mass(prior, state, ir, povm, b)
                won = _mass(prior, state, accept, povm, b)
                if not 0 <= won <= p <= 1:
                    raise ArithmeticError("Invalid transcript probability after exact validation")
                total = exact_scalar(total + p, max_bits=b.max_scalar_bits)
                score = exact_scalar(score + won, max_bits=b.max_scalar_bits)
                transcripts.append(Transcript(index, h, z, p, won))
    if total != 1:
        raise ArithmeticError("Transcript mass is not one after exact validation")
    return ReplayResult(score, tuple(transcripts))
