"""Separated dispatch, fixed wires, exact steering and admission for Alice-then-Bob."""
from dataclasses import replace
from fractions import Fraction as F
from importlib import import_module
from itertools import product

import pytest

from nonlocalgames.protocols.interactive import (
    BobMove, Continuation, FinalMove, FirstMove, KrausTerm, LocalHistory, PrivateCase,
    ReplayLimits, TwoProverProtocol, TwoProverStrategy, replay, replay_two_prover,
)
from nonlocalgames.protocols.interactive.examples import split_bell_label, two_prover_bit_relay
from nonlocalgames.quantum.exact import ExactBudgetExceeded, RationalMatrix as M
from nonlocalgames.quantum.exact.algebra import identity, scale, subtract


def test_bob_only_sees_the_delivered_bit_and_intentional_relay_changes_only_the_verifier():
    hidden, s = two_prover_bit_relay()
    relay, same = two_prover_bit_relay(forward_answer=True)
    assert s == same
    for forward, p in ((False, hidden), (True, relay)):
        result = replay_two_prover(p, s)
        assert result.acceptance_probability == (1 if forward else F(1, 2))
        assert len(result.transcripts) == 16
        for leaf, (case, y, z) in zip(result.transcripts, product(range(4), range(2), range(2)),
                                       strict=True):
            x, coin = divmod(case, 2)
            v = y if forward else coin
            assert (leaf.private_case, leaf.alice_question, leaf.alice_answer,
                    leaf.bob_question, leaf.bob_answer) == (case, x, y, v, z)
            prob = F(y == x and z == v, 4)
            assert (leaf.probability, leaf.accepted_mass) == (prob, prob if z == x else 0)
        # The actual classical messages to Bob have the same marginal under both verifiers.
        assert sum(t.probability for t in result.transcripts if t.bob_question == 1) == F(1, 2)
    # Each v in the hidden verifier occurs across different x, y and private cases.
    leaves = replay_two_prover(hidden, s).transcripts
    for v in range(2):
        assert {(t.alice_question, t.alice_answer) for t in leaves
                if t.bob_question == v} == set(product(range(2), repeat=2))


def test_input_correlations_steer_bob_without_passing_any_alice_memory_or_classical_message():
    i1 = identity(1)
    e0, e1 = M(((1, 0), (0, 0))), M(((0, 0), (0, 1)))
    # Read Z, keeping Alice's output in |0> regardless of y. Bob receives constant v.
    alice = FirstMove(4, ((KrausTerm(1, e0),), (KrausTerm(1, M(((0, 1), (0, 0)))),)))
    s = TwoProverStrategy((alice,), (BobMove(8, (e0, e1)),))
    tests = (i1, scale(i1, 0))
    follow = (Continuation(8, tests), Continuation(8, tests[::-1]))
    # Q_B,Q_A order; amplitudes 3/5 and 4/5. Start with identical individual
    # marginals in both variants but destroy correlations in the product control.
    rho = scale(M(((9, 0, 0, 12), (0, 0, 0, 0), (0, 0, 0, 0), (12, 0, 0, 16))), F(1, 25))
    product_rho = scale(M(((81, 0, 0, 0), (0, 144, 0, 0), (0, 0, 144, 0),
                           (0, 0, 0, 256))), F(1, 625))
    p = TwoProverProtocol(reference_dimension=1, alice_dimension=2, bob_dimension=2,
                          cases=(PrivateCase(1, 4, rho, follow),))
    result = replay_two_prover(p, s)
    assert [t.probability for t in result.transcripts] == [F(9, 25), 0, 0, F(16, 25)]
    assert result.acceptance_probability == 1
    # Keep |y> instead of resetting to |0>: Alice's residual register is never sent.
    keep = FirstMove(4, tuple((KrausTerm(1, op),) for op in (e0, e1)))
    assert replay_two_prover(p, replace(s, alice=(keep,))) == result
    control = replay_two_prover(replace(p, cases=(replace(p.cases[0], state=product_rho),)), s)
    assert control.acceptance_probability == F(337, 625)
    assert [t.probability for t in control.transcripts] == [F(k, 625) for k in (81, 144, 144, 256)]
    # Changing the instrument's output unitaries must not turn a residual wire over to Bob.
    keep = FirstMove(4, ((KrausTerm(1, e0),), (KrausTerm(1, e1),)))
    assert replay_two_prover(p, replace(s, alice=(keep,))) == result


def test_acknowledgement_does_not_allow_bob_to_make_a_joint_bell_measurement():
    p, one = split_bell_label()
    assert replay(p, one).acceptance_probability == 1
    split = TwoProverProtocol(reference_dimension=1, alice_dimension=2, bob_dimension=2,
                              cases=p.cases)
    bob = BobMove(0, (scale(identity(2), F(1, 4)),)*4)
    s = TwoProverStrategy(one.first, (bob,))
    result = replay_two_prover(split, s)
    assert result.acceptance_probability == F(1, 4)
    assert {t.probability for t in result.transcripts} == {F(1, 16)}
    with pytest.raises(ValueError, match="Matrix dimension"):
        replay_two_prover(split, replace(s, bob=(BobMove(0, one.final[0].effects),)))


def test_unreceived_metadata_cannot_select_a_bob_move_or_an_alice_move():
    p, s = two_prover_bit_relay()
    with pytest.raises(TypeError, match="TwoProverStrategy"):
        replay_two_prover(p, lambda case, x, y, v: s)
    with pytest.raises(TypeError, match="LocalStrategy"):
        replay(p, s)
    with pytest.raises(TypeError, match="BobMove"):
        replay_two_prover(p, replace(s, bob=(FinalMove(LocalHistory(0, 0, 0), s.bob[0].effects),)))
    for key in ((0, 0, 0), LocalHistory(0, 0, 0), True, -1, 2**31):
        with pytest.raises(ValueError, match="Classical labels"):
            replay_two_prover(p, replace(s, bob=(replace(s.bob[0], question=key), *s.bob[1:])))
    with pytest.raises(ValueError, match="Duplicate Bob question"):
        replay_two_prover(p, replace(s, bob=s.bob + (replace(s.bob[0], effects=s.bob[1].effects),)))
    with pytest.raises(ValueError, match="unused moves"):
        replay_two_prover(p, replace(s, bob=s.bob + (replace(s.bob[0], question=3),)))
    with pytest.raises(ValueError, match="Missing final move"):
        replay_two_prover(p, replace(s, bob=s.bob[:1]))
    with pytest.raises(ValueError, match="Duplicate first question"):
        replay_two_prover(p, replace(s, alice=s.alice*2))
    with pytest.raises(ValueError, match="unused moves"):
        replay_two_prover(p, replace(s, alice=s.alice + (replace(s.alice[0], question=3),)))
    with pytest.raises(TypeError, match="tuple"):
        replay_two_prover(p, replace(s, bob=iter(s.bob)))


def test_bob_alphabet_and_physics_enforced_even_on_impossible_rows():
    p, s = two_prover_bit_relay()
    c = p.cases[0]  # this case has x=0 and never emits y=1
    cont = c.after_first[1]
    assert cont.question == c.after_first[0].question == 0
    for variant, expected in [
        (replace(cont, accept=cont.accept[:1]), "final-answer alphabet"),
        (replace(cont, question=6), "Missing final move"),
        (replace(cont, accept=(M(((2,),)), M(((0,),)))), "not PSD"),
    ]:
        case = replace(c, after_first=(c.after_first[0], variant))
        with pytest.raises(ValueError, match=expected):
            replay_two_prover(replace(p, cases=(case, *p.cases[1:])), s)
    p = replace(p, cases=(replace(c, after_first=(c.after_first[0], replace(cont, question=6))),
                          *p.cases[1:]))
    for effects, reason in [((M(((2,),)), M(((-1,),))), "not PSD"),
                            ((identity(1), identity(1)), "sum to identity")]:
        with pytest.raises(ValueError, match=reason):
            replay_two_prover(p, replace(s, bob=(*s.bob, BobMove(6, effects))))
    with pytest.raises(ValueError, match="trace preserving"):
        replay_two_prover(p, replace(s, alice=(replace(s.alice[0], outcomes=((), ())), s.alice[1]),
                                     bob=(*s.bob, BobMove(6, s.bob[0].effects))))


def _budget_fixture():
    # Two private cases reuse both Alice outcomes, even the impossible y=1,
    # and use a *single* B POVM across all four verifier continuations.
    i2, i3, i1 = identity(2), identity(3), identity(1)
    terms = (KrausTerm(F(2, 3), i2), KrausTerm(F(1, 3), i2))
    b0 = M(((1, 0, 0), (0, 0, 0), (0, 0, 0)))
    bob = BobMove(11, (b0, subtract(i3, b0)))
    s = TwoProverStrategy((FirstMove(9, (terms, ())),), (bob,))
    rho = M(tuple(tuple(int(i == j == 0) for j in range(6)) for i in range(6)))
    follow = (Continuation(11, (i1, scale(i1, 0))),)*2
    cases = tuple(PrivateCase(prior, 9, rho, follow) for prior in (F(1, 4), F(3, 4)))
    return TwoProverProtocol(reference_dimension=1, alice_dimension=2, bob_dimension=3,
                             cases=cases), s


def test_factorized_budget_boundaries_precede_dense_allocation(monkeypatch):
    p, s = _budget_fixture()
    # Inputs: 2*4 Kraus, 2*9 B effects (not per history), 2*36 states, 8 R tests.
    # Work: 2*4 A-outcome sums, 2*20 Kraus, 2*225 B-PSD/sums, 2*1728 states;
    # per leaf: 36 for B⊗A, 2*8+1 for R tests, 4*36 for two lifts and traces.
    defaults = ReplayLimits()
    b = replace(defaults.exact, max_joint_dimension=6, max_psd_size=12, max_events=8,
                max_matrix_entries=106, max_scalar_bits=3, max_parties=2)
    limits = replace(defaults, exact=b, max_kraus_terms=2, max_dense_work=5530)
    assert replay_two_prover(p, s, limits=limits).acceptance_probability == 1
    module = import_module("nonlocalgames.protocols.interactive.two_prover")
    monkeypatch.setattr(module, "identity", lambda *a, **kw: pytest.fail("allocated before admission"))
    for small, reason in [
        (replace(limits, exact=replace(b, max_joint_dimension=5)), "max_joint_dimension"),
        (replace(limits, exact=replace(b, max_psd_size=11)), "max_psd_size"),
        (replace(limits, exact=replace(b, max_events=7)), "transcript leaves"),
        (replace(limits, exact=replace(b, max_matrix_entries=105)), "total matrix entries"),
        (replace(limits, exact=replace(b, max_scalar_bits=2)), "max_scalar_bits"),
        (replace(limits, exact=replace(b, max_parties=1)), "max_parties"),
        (replace(limits, max_kraus_terms=1), "Kraus terms"),
        (replace(limits, max_dense_work=5529), "max_dense_work"),
    ]:
        with pytest.raises(ExactBudgetExceeded, match=reason):
            replay_two_prover(p, s, limits=small)
    for bad in (True, F(3, 2), -1, 0):
        with pytest.raises(ValueError, match="positive integers"):
            replay_two_prover(replace(p, bob_dimension=bad), s)
    with pytest.raises(TypeError, match="ReplayLimits"):
        replay_two_prover(p, s, limits=None)
    with pytest.raises(TypeError, match="RationalMatrix"):
        replay_two_prover(p, replace(s, bob=(replace(s.bob[0], effects=(None, None)),)))
    with pytest.raises(ValueError, match="Matrix dimension"):
        replay_two_prover(p, replace(s, bob=(replace(s.bob[0], effects=(identity(6),)*2),)))
    c = p.cases[0]
    wrong = replace(c.after_first[1], accept=(identity(3),)*2)
    with pytest.raises(ValueError, match="Matrix dimension"):
        replay_two_prover(replace(p, cases=(replace(c, after_first=(c.after_first[0], wrong)),
                                            *p.cases[1:])), s)
