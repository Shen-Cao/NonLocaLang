"""Transfer a prepared second register across the cut, not a description or a fresh state."""
from dataclasses import replace
from fractions import Fraction as F
from importlib import import_module
from itertools import product

import pytest

from nonlocalgames.protocols.interactive import (
    Continuation, FinalMove, FirstMove, KrausTerm, LocalHistory, LocalStrategy, PrivateCase,
    ReplayLimits, TwoRoundProtocol, replay,
)
from nonlocalgames.protocols.interactive.examples import sign_discrimination, split_bell_label
from nonlocalgames.quantum.exact import ExactBudget, ExactBudgetExceeded, RationalMatrix as M
from nonlocalgames.quantum.exact.algebra import identity, scale, subtract


@pytest.mark.parametrize("memory, independent, score", [
    ("retain", False, 1), ("dephase", False, F(1, 2)),
    ("reset", False, F(1, 4)), ("retain", True, F(1, 4)),
])
def test_private_bell_label_requires_the_transferred_correlations(memory, independent, score):
    protocol, strategy = split_bell_label(memory, independent=independent)
    assert (protocol.reference_dimension, protocol.second_message_dimension,
            protocol.message_dimension) == (1, 2, 2)
    for case in protocol.cases:
        assert case.probability == F(1, 4) and case.question == case.after_first[0].question == 0
        # Both individual wires are exactly I/2 in every case, including the control.
        for i, j in product(range(2), repeat=2):
            for grid, expected in ((case.state.real, F(i == j, 2)), (case.state.imag, 0)):
                assert sum(grid[2*i+k][2*j+k] for k in range(2)) == expected
                assert sum(grid[2*k+i][2*k+j] for k in range(2)) == expected
    report = replay(protocol, strategy)
    assert report.acceptance_probability == score
    assert len(report.transcripts) == 16
    assert {t.history for t in report.transcripts} == {LocalHistory(0, 0, 0)}
    for t in report.transcripts:
        label, z = t.private_case, t.final_answer
        if independent or memory == "reset":
            conditional = F(1, 4)
        elif memory == "dephase":
            conditional = F(label//2 == z//2, 2)  # learn parity, but not phase
        else:
            conditional = int(label == z)
        assert t.probability == F(1, 4)*conditional
        assert t.accepted_mass == (t.probability if label == z else 0)


def test_memory_controls_change_only_the_instrument_and_marginal_control_only_the_preparation():
    p, s = split_bell_label()
    for label, case in enumerate(p.cases):
        a, b = (0, 3) if label < 2 else (1, 2)
        expected = _pair_projector(4, a, b, (1 if label % 2 == 0 else -1, 0))
        assert case.state == s.final[0].effects[label] == expected
    for memory in ("dephase", "reset"):
        p2, s2 = split_bell_label(memory)
        assert p2 == p and s2.first != s.first and s2.final == s.final
    marginal, unchanged = split_bell_label(independent=True)
    product_state = scale(identity(4), F(1, 4))
    assert unchanged == s
    assert marginal.cases == tuple(replace(c, state=product_state) for c in p.cases)
    assert all(c.state != product_state for c in p.cases)


def _pair_projector(dim, a, b, off_diagonal, *, budget=None):
    """Two equal populations with <a|P|b> = (re + i im)/2; no tensor algebra."""
    re, im = off_diagonal
    real = [[0]*dim for _ in range(dim)]
    imag = [[0]*dim for _ in range(dim)]
    real[a][a] = real[b][b] = F(1, 2)
    real[a][b] = real[b][a] = F(re, 2)
    imag[a][b], imag[b][a] = F(im, 2), F(-im, 2)
    return M(real, imag, budget=budget)


def test_asymmetric_complex_branch_oracle_with_three_unequal_nontrivial_factors():
    # R_keep, Q2, Q1 sizes 2,3,4. Flatten |0,1,1> and |1,2,3> to 5 and 23.
    limits = ReplayLimits(exact=ExactBudget(max_joint_dimension=24, max_psd_size=48))
    rho = _pair_projector(24, 5, 23, (1, 0), budget=limits.exact)
    # K± sends |1> -> ±i|2>, |3> -> |0>. Thus the two populated final Q2,M
    # kets are |1,2>, |2,0>, i.e. indices 6,8. Project onto (|6>-i|8>)/sqrt(2):
    # R_keep collapses to X+ for y=0, X− for y=1 (opposite for the complement).
    k_real = ((0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0))
    outcomes = tuple((KrausTerm(weight, M(k_real, ((0,)*4, (0,)*4, (0, sign, 0, 0), (0,)*4))),)
                     for weight, sign in ((F(1, 4), 1), (F(3, 4), -1)))
    local = _pair_projector(12, 6, 8, (0, 1))
    povm = (local, subtract(identity(12), local))
    xplus = _pair_projector(2, 0, 1, (1, 0))
    xminus = subtract(identity(2), xplus)
    follow = (Continuation(17, (xplus, xminus)), Continuation(17, (xminus, xplus)))
    protocol = TwoRoundProtocol(2, 4, (PrivateCase(1, 11, rho, follow),), second_message_dimension=3)
    strategy = LocalStrategy((FirstMove(11, outcomes),),
                             tuple(FinalMove(LocalHistory(11, y, 17), povm) for y in range(2)))
    report = replay(protocol, strategy, limits=limits)
    assert report.acceptance_probability == 1
    expected = [F(1, 8), F(1, 8), F(3, 8), F(3, 8)]
    assert [t.probability for t in report.transcripts] == expected
    assert [t.accepted_mass for t in report.transcripts] == expected


def test_destructive_first_readout_steers_the_later_wire_with_unequal_branch_masses():
    # Initial ket (4|0,1> + 3|1,0>)/5 in Q2,Q1 order. Read Z on Q1, reply y,
    # and reset M to |0>. Q2 must arrive as |1-y>, not its average diagonal state.
    rho = scale(M(((0, 0, 0, 0), (0, 16, 12, 0), (0, 12, 9, 0), (0, 0, 0, 0))), F(1, 25))
    one, zero = identity(1), M(((0,),))
    cont = (Continuation(0, (zero, one)), Continuation(0, (one, zero)))
    p = TwoRoundProtocol(1, 2, (PrivateCase(1, 0, rho, cont),), second_message_dimension=2)
    outcomes = tuple((KrausTerm(1, M(((1-y, y), (0, 0)))),) for y in range(2))
    # Final POVM reads Q2 only; no use of y or M in this supplied readout.
    q2_zero = M(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)))
    effects = (q2_zero, subtract(identity(4), q2_zero))
    s = LocalStrategy((FirstMove(0, outcomes),),
                      tuple(FinalMove(LocalHistory(0, y, 0), effects) for y in range(2)))
    report = replay(p, s)
    assert report.acceptance_probability == 1
    assert [t.probability for t in report.transcripts] == [0, F(9, 25), F(16, 25), 0]


def test_classical_only_followup_is_the_legacy_constructor_including_transcript():
    p, s = sign_discrimination()
    assert p.second_message_dimension == 1
    explicit = TwoRoundProtocol(p.reference_dimension, p.message_dimension, p.cases,
                                second_message_dimension=1)
    assert replay(explicit, s) == replay(p, s)
    assert [t.probability for t in replay(explicit, s).transcripts] == [
        F(9, 20), F(1, 20), F(1, 20), F(9, 20),
    ]


def test_each_operation_admits_only_the_wires_owned_at_its_move():
    p, s = split_bell_label()
    with pytest.raises(ValueError, match="Matrix dimension"):
        replay(p, replace(s, first=(FirstMove(0, ((KrausTerm(1, identity(4)),),)),)))  # too early
    with pytest.raises(ValueError, match="Matrix dimension"):
        replay(p, replace(s, final=(replace(s.final[0], effects=(identity(2),)*4),)))
    cont = replace(p.cases[0].after_first[0], accept=(identity(2),)*4)  # Q2 already sent
    with pytest.raises(ValueError, match="Matrix dimension"):
        replay(replace(p, cases=(replace(p.cases[0], after_first=(cont,)), *p.cases[1:])), s)
    # The private label cannot select an undisclosed first question or final history.
    with pytest.raises(ValueError, match="unused moves"):
        replay(p, replace(s, first=(*s.first, replace(s.first[0], question=1))))
    with pytest.raises(ValueError, match="unused moves"):
        replay(p, replace(s, final=(*s.final, FinalMove(LocalHistory(0, 0, 1), s.final[0].effects))))


def test_impossible_transfer_branch_checked_for_joint_final_and_reference_effects():
    p, s = split_bell_label()
    p = replace(p, cases=tuple(replace(c, after_first=c.after_first*2) for c in p.cases))
    unreachable = FinalMove(LocalHistory(0, 1, 0), s.final[0].effects)
    s = replace(s, first=(replace(s.first[0], outcomes=(*s.first[0].outcomes, ())),),
                final=(*s.final, unreachable))
    leaves = [t for t in replay(p, s).transcripts if t.history.first_answer == 1]
    assert len(leaves) == 16 and all(t.probability == t.accepted_mass == 0 for t in leaves)
    i = identity(4)
    bad = replace(unreachable, effects=tuple(scale(i, n) for n in (-1, 2, 0, 0)))
    with pytest.raises(ValueError, match="not PSD"):
        replay(p, replace(s, final=(s.final[0], bad)))
    c = p.cases[0]
    cont = replace(c.after_first[1], accept=(M(((2,),)), *c.after_first[1].accept[1:]))
    with pytest.raises(ValueError, match="not PSD"):
        replay(replace(p, cases=(replace(c, after_first=(c.after_first[0], cont)), *p.cases[1:])), s)


def test_second_wire_budgets_include_joint_preparations_and_final_povm_before_dense_allocation(
        monkeypatch):
    p, s = split_bell_label()
    # Entries: one 2x2 Kraus operator, four 4x4 POVM effects, four 4x4 states,
    # sixteen scalar tests. Work: 20 Kraus + 4*528 final + 4*(512+208+4*81) replay.
    exact = ExactBudget(max_joint_dimension=4, max_psd_size=8, max_matrix_entries=148, max_events=16)
    limits = ReplayLimits(exact=exact, max_kraus_terms=1, max_dense_work=6308)
    assert replay(p, s, limits=limits).acceptance_probability == 1
    module = import_module("nonlocalgames.protocols.interactive.replay")
    monkeypatch.setattr(module, "identity", lambda *a, **kw: pytest.fail("allocated before admission"))
    for smaller, reason in [
        (replace(limits, exact=replace(exact, max_joint_dimension=3)), "max_joint_dimension"),
        (replace(limits, exact=replace(exact, max_psd_size=7)), "max_psd_size"),
        (replace(limits, exact=replace(exact, max_matrix_entries=147)), "total matrix entries"),
        (replace(limits, exact=replace(exact, max_events=15)), "transcript leaves"),
        (replace(limits, max_dense_work=6307), "max_dense_work"),
    ]:
        with pytest.raises(ExactBudgetExceeded, match=reason):
            replay(p, s, limits=smaller)
    for bad_dim in (True, 0, -1, F(3, 2)):
        with pytest.raises(ValueError, match="positive integers"):
            replay(replace(p, second_message_dimension=bad_dim), s)
