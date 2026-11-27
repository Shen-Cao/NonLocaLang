"""Exact point evaluation with one local turn per prover, Alice before Bob."""
from fractions import Fraction

from ...quantum.exact.algebra import identity, kron
from ...quantum.exact.scalars import exact_scalar
from ._plan import prepare_two_prover
from .model import (
    ReplayLimits, TwoProverProtocol, TwoProverResult, TwoProverStrategy, TwoProverTranscript,
)
from .replay import _check_physics, _mass


def replay_two_prover(protocol: TwoProverProtocol, strategy: TwoProverStrategy, *,
                      limits: ReplayLimits = ReplayLimits()) -> TwoProverResult:
    """Check all declared data, then enumerate all leaves including impossible ones.

    A[x,y] = Σ_k w K†K acts on Q_A. No operation ever reads Alice's residual
    memory. Bob's POVM B[v] acts only on Q_B; the case and y can select v through
    the verifier's table, but only this delivered v selects Bob's row.
    """
    (r, qb, qa), cases, first, bob = prepare_two_prover(protocol, strategy, limits)
    b = limits.exact
    ir, ia, ib = (identity(n, budget=b) for n in (r, qa, qb))
    alice = _check_physics(cases, first, bob, ir, ia, ib, b, keep_outcome_effects=True)

    total = score = Fraction(0)
    transcripts = []
    for index, (prior, case) in enumerate(cases):
        x = case.question
        for y, (ay, cont) in enumerate(zip(alice[x], case.after_first, strict=True)):
            v = cont.question
            for z, (bz, accept) in enumerate(zip(bob[v], cont.accept, strict=True)):
                ba = kron(bz, ay, budget=b)
                p = _mass(prior, case.state, ir, ba, b)
                won = _mass(prior, case.state, accept, ba, b)
                if not 0 <= won <= p <= 1:
                    raise ArithmeticError("Invalid transcript probability after exact validation")
                total = exact_scalar(total + p, max_bits=b.max_scalar_bits)
                score = exact_scalar(score + won, max_bits=b.max_scalar_bits)
                transcripts.append(TwoProverTranscript(index, x, y, v, z, p, won))
    if total != 1:
        raise ArithmeticError("Transcript mass is not one after exact validation")
    return TwoProverResult(score, tuple(transcripts))
