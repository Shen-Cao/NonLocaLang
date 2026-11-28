"""Independent forward Schrödinger oracle for the separated Heisenberg contraction."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

from sympy import I, Matrix, Rational as Q, eye, im, re, zeros

from nonlocalgames.protocols.interactive import (
    BobMove, Continuation, FirstMove, KrausTerm, PrivateCase, ReplayLimits,
    TwoProverProtocol, TwoProverStrategy, replay_two_prover,
)
from nonlocalgames.quantum.exact import RationalMatrix


def _matrix(m, budget=None):
    def grid(component):
        return tuple(tuple(F(component(m[i, j].expand())) for j in range(m.cols)) for i in range(m.rows))
    return RationalMatrix(grid(re), grid(im), budget=budget)


def _ket_norm(ket):
    return sum(abs(a)**2 for a in ket.values())


def _preparation(ensemble):
    rho = zeros(24, 24)
    for weight, ket in ensemble:
        psi = zeros(24, 1)
        for (r, b, a), amp in ket.items():
            psi[(r*3+b)*4+a] = amp
        rho += weight*psi*psi.H/_ket_norm(ket)
    return rho.applyfunc(lambda e: e.expand())


def _alice_instrument(sign):
    def reset(m, entries):
        k = zeros(4, 4)
        for a, amplitude in entries.items():
            k[m, a] = amplitude
        return k

    # Two orthogonal complex bras on {1,3}. Other basis inputs have different
    # response probabilities, and several non-Hermitian Kraus terms reset memory.
    a0 = reset(0, {1: Q(3, 5), 3: -sign*I*Q(4, 5)})
    a1 = reset(2, {1: Q(4, 5), 3: sign*I*Q(3, 5)})
    return (((1, a0), (Q(1, 3), reset(1, {2: 1}))),
            ((1, a1), (Q(2, 3), reset(0, {2: 1})), (1, reset(3, {0: 1}))))


def _forward(ensemble, terms):
    """Push each ket through each K on Alice; trace out her output by explicit index."""
    tau = zeros(6, 6)  # only R and B remain, before Bob's measurement
    for w, ket in ensemble:
        for t, k in terms:
            # Enumerate A's *output* columns; only equal columns interfere after discard.
            output = zeros(6, 4)
            for (r, b, a), amp in ket.items():
                for m in range(4):
                    output[r*3+b, m] += k[m, a]*amp
            tau += w*t*output*output.H/_ket_norm(ket)
    return tau.applyfunc(lambda e: e.expand())


def _expect(tau, ref, bob):
    return F(sum(ref[r, s]*bob[b, c]*tau[s*3+c, r*3+b]
                          for r, s in product(range(2), repeat=2)
                          for b, c in product(range(3), repeat=2)).expand())


def test_each_mass_against_forward_oracle_with_private_mixtures_and_unequal_complex_registers():
    # Register dimensions R=2, B=3, A=4; private mixed states and effects are
    # not jointly real/diagonal. The same v and same x occur in different cases.
    ket0 = {(0, 0, 1): 1+I, (0, 2, 3): 2, (1, 0, 3): 1-I, (1, 2, 1): I,
         (1, 1, 0): 1, (0, 1, 2): 1-I}
    ket1 = {(0, 2, 1): 2-I, (0, 0, 3): I, (1, 2, 3): -1, (0, 1, 0): 1,
         (1, 0, 1): I, (1, 1, 2): 1+I}
    ensembles = (((Q(2, 5), ket0), (Q(3, 5), ket1)), ((Q(4, 7), ket0), (Q(3, 7), ket1)),
                 ((1, ket1),))
    # x=7 declares an impossible third outcome; all its Bob/acceptance data are checked.
    instruments = {7: (*_alice_instrument(1), ()), 13: _alice_instrument(-1)}
    priors = (F(1, 5), F(3, 10), F(1, 2))
    dispatch = ((7, (11, 13, 13)), (13, (11, 11)), (7, (11, 11, 13)))
    psi = Matrix([3, 0, 4*I])/5
    b0 = psi*psi.H
    b0[1, 1] = Q(1, 3)
    phi = Matrix([0, 5*I, 12])/13
    b1 = phi*phi.H
    b1[0, 0] = Q(2, 5)
    bob = {11: (b0, eye(3)-b0), 13: (b1, eye(3)-b1, zeros(3, 3))}
    ref = (Matrix([[2, -1-I], [-1+I, 3]])/5,
           Matrix([[9, 12*I], [-12*I, 16]])/25, eye(2))
    limits = ReplayLimits()
    limits = replace(limits, exact=replace(limits.exact, max_joint_dimension=24, max_psd_size=48))
    alice = tuple(FirstMove(x, tuple(tuple(KrausTerm(F(w), _matrix(op)) for w, op in group)
                                     for group in groups)) for x, groups in instruments.items())
    strategy = TwoProverStrategy(alice, tuple(BobMove(v, tuple(map(_matrix, effects)))
                                              for v, effects in bob.items()))
    expected, cases = [], []
    catches_alice_conjugation = catches_bob_conjugation = catches_reference_conjugation = False
    for n, (prior, ensemble, (x, vs)) in enumerate(zip(priors, ensembles, dispatch, strict=True)):
        continuations = []
        for y, v in enumerate(vs):
            terms = instruments[x][y]
            tau = _forward(ensemble, terms)
            conjugated = _forward(ensemble, tuple((w, op.conjugate()) for w, op in terms))
            tests = []
            for z, bz in enumerate(bob[v]):
                ez = ref[(n+y+z) % len(ref)]
                p, won = prior*_expect(tau, eye(2), bz), prior*_expect(tau, ez, bz)
                catches_alice_conjugation |= p != prior*_expect(conjugated, eye(2), bz)
                catches_bob_conjugation |= p != prior*_expect(tau, eye(2), bz.conjugate())
                catches_reference_conjugation |= won != prior*_expect(tau, ez.conjugate(), bz)
                expected.append((n, x, y, v, z, p, won))
                tests.append(_matrix(ez))
            continuations.append(Continuation(v, tuple(tests)))
        cases.append(PrivateCase(prior, x, _matrix(_preparation(ensemble), limits.exact),
                                 tuple(continuations)))
    protocol = TwoProverProtocol(reference_dimension=2, alice_dimension=4, bob_dimension=3,
                                 cases=tuple(cases))
    report = replay_two_prover(protocol, strategy, limits=limits)
    actual = [(t.private_case, t.alice_question, t.alice_answer, t.bob_question, t.bob_answer,
               t.probability, t.accepted_mass) for t in report.transcripts]
    assert catches_alice_conjugation and catches_bob_conjugation and catches_reference_conjugation
    assert len(expected) == 19 and sum(row[-2] == 0 for row in expected) == 7
    # The one registered Bob row at v=11 covers every realized (x,y) below;
    # even holding all three questions/answers x,y,v fixed can leave the case hidden.
    assert {(t[0], t[1], t[2]) for t in expected if t[3] == 11 and t[-2]} == {
        (0, 7, 0), (1, 13, 0), (1, 13, 1), (2, 7, 0), (2, 7, 1),
    }
    assert actual == expected
    assert sum(t[-2] for t in expected) == 1
    assert report.acceptance_probability == sum(t[-1] for t in expected) == F(34398248459, 54898593750)
