"""Independent small controls for causal dispatch, memory, and exact probabilities."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

from nonlocalgames.protocols.interactive import (
    Continuation, FinalMove, FirstMove, KrausTerm, LocalHistory, LocalStrategy,
    PrivateCase, TwoRoundProtocol, replay,
)
from nonlocalgames.protocols.interactive.examples import (
    correlated_memory, delayed_basis, sign_discrimination,
)
from nonlocalgames.quantum.exact import RationalMatrix as M
from nonlocalgames.quantum.exact.algebra import identity, kron, scale, subtract


def _scalar(x):
    return M(((x,),))


def test_private_sign_matches_independent_two_outcome_upper_bound():
    protocol, strategy = sign_discrimination()
    report = replay(protocol, strategy)
    assert len(report.transcripts) == 4
    assert all(t.history == LocalHistory(0, 0, 0) for t in report.transcripts)
    assert [t.probability for t in report.transcripts] == [F(9, 20), F(1, 20), F(1, 20), F(9, 20)]
    # Exact weak dual, not a replay status. For each private sign, verify Y-p_s*rho_s
    # is real symmetric PSD directly by its two diagonal minors and determinant.
    y0, y1 = F(3, 5), F(3, 10)
    for case, s in zip(protocol.cases, (1, -1), strict=True):
        assert case.probability == F(1, 2)
        assert case.state.real == ((F(4, 5), F(2*s, 5)), (F(2*s, 5), F(1, 5)))
        assert case.state.imag == ((0, 0), (0, 0))
        a, b = case.state.real[0]
        _, d = case.state.real[1]
        a, b, d = y0-a/2, -b/2, y1-d/2
        assert a == d == F(1, 5) and a*d == b*b
    assert report.acceptance_probability == y0+y1 == F(9, 10)
    assert sum(t.accepted_mass for t in report.transcripts) == report.acceptance_probability


def test_revealing_a_classical_bit_and_dephasing_the_qubit_are_different_operations():
    private, _ = sign_discrimination()
    exposed, informed = sign_discrimination(reveal=True)
    dephased, blind = sign_discrimination(dephase=True)
    assert [c.state for c in private.cases] == [c.state for c in exposed.cases]
    assert [c.question for c in private.cases] == [0, 0]
    assert [c.question for c in exposed.cases] == [0, 1]
    assert dephased.cases[0].state == dephased.cases[1].state
    assert dephased.cases[0].state.real == ((F(4, 5), 0), (0, F(1, 5)))
    assert replay(exposed, informed).acceptance_probability == 1
    assert replay(dephased, blind).acceptance_probability == F(1, 2)


def test_same_first_bit_law_and_same_average_channel_do_not_fix_transcript_score():
    expected = {"matched": F(1), "swapped": F(0), "independent": F(1, 2)}
    verifier, base = correlated_memory()
    for mode, score in expected.items():
        protocol, strategy = correlated_memory(mode)
        assert protocol == verifier and strategy.final == base.final
        groups = strategy.first[0].outcomes
        # Independently check each entry of the average channel on every E_ij.
        # All example Kraus operators are real and diagonal. This is precisely
        # computational-basis dephasing, not just equality on this one input.
        for i, j, row, col in product(range(2), repeat=4):
            coefficient = sum(F(t.weight)*t.operator.real[row][i]*t.operator.real[col][j]
                              for group in groups for t in group)
            assert coefficient == int(i == j == row == col)
        report = replay(protocol, strategy)
        assert [(t.history.first_answer, t.final_answer, t.probability) for t in report.transcripts
                ] == [(y, z, F(1, 4)) for y, z in product(range(2), repeat=2)]
        assert report.acceptance_probability == score


def test_adaptive_verifier_reveals_private_bit_xor_first_answer_only_after_commitment():
    # Independent private bit s and truthful first output y; only later is v=s XOR y sent.
    one = _scalar(1)
    coin = (KrausTerm(F(1, 2), one),)
    cases = tuple(PrivateCase(F(1, 2), 0, one, tuple(
        Continuation(s ^ y, tuple(_scalar(int(s == z)) for z in range(2))) for y in range(2)))
        for s in range(2))
    protocol = TwoRoundProtocol(1, 1, cases)
    first = (FirstMove(0, (coin, coin)),)

    def policy(use_earlier_bit):
        return LocalStrategy(first, tuple(FinalMove(LocalHistory(0, y, v), tuple(
            _scalar(int(z == (v ^ y if use_earlier_bit else v))) for z in range(2)))
            for y, v in product(range(2), repeat=2)))

    report = replay(protocol, policy(True))
    assert len(report.transcripts) == 8  # includes impossible final answers
    for t in report.transcripts:
        s, h = t.private_case, t.history
        assert h.second_question == s ^ h.first_answer
        assert t.probability == t.accepted_mass == (F(1, 4) if s == t.final_answer else 0)
    assert report.acceptance_probability == 1
    assert replay(protocol, policy(False)).acceptance_probability == F(1, 2)


def test_delayed_basis_preserves_a_qubit_or_restricts_memory_by_actually_resetting_it():
    protocol, memory = delayed_basis()
    other, reset = delayed_basis(keep_qubit=False)
    assert other == protocol
    report = replay(protocol, memory)
    assert report.acceptance_probability == 1
    assert len(report.transcripts) == 16
    assert all(t.probability == 0 for t in report.transcripts if t.history.first_answer == 1)
    assert sum(t.probability for t in report.transcripts) == 1
    assert replay(protocol, reset).acceptance_probability == F(25, 26)


def test_revealing_the_basis_early_changes_the_protocol_and_closes_the_classical_memory_gap():
    protocol, _ = delayed_basis(keep_qubit=False)
    early = replace(protocol, cases=tuple(replace(c, question=c.after_first[0].question)
                                         for c in protocol.cases))
    rows = []
    for b in range(2):
        s = 1-2*b
        # Measure the disclosed basis, store bit y, then leave memory in |0>.
        rows.append(FirstMove(b, tuple((KrausTerm(F(1, 26), M((bra, (0, 0)))),)
                                       for bra in ((5, s), (-s, 5)))))
    reads = tuple(FinalMove(LocalHistory(b, y, b), tuple(scale(identity(2), int(y == z))
                                                         for z in range(2)))
                  for b, y in product(range(2), repeat=2))
    assert replay(early, LocalStrategy(tuple(rows), reads)).acceptance_probability == 1


def test_complex_nonhermitian_kraus_and_unequal_R_Q_factor_order():
    r1 = M(((0, 0), (0, 1)))
    xplus = scale(M(((1, 1, 0), (1, 1, 0), (0, 0, 0))), F(1, 2))
    phase = M(((1, 0, 0), (0, 0, 0), (0, 0, 1)), ((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    yplus = scale(M(((1, 0, 0), (0, 1, 0), (0, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 0))),
                  F(1, 2))
    cont = Continuation(0, (r1, scale(r1, 0)))
    protocol = TwoRoundProtocol(2, 3, (PrivateCase(1, 0, kron(r1, xplus), (cont,)),))
    strategy = LocalStrategy((FirstMove(0, ((KrausTerm(1, phase),),)),),
                             (FinalMove(LocalHistory(0, 0, 0),
                                        (yplus, subtract(identity(3), yplus))),))
    assert replay(protocol, strategy).acceptance_probability == 1


def test_terminal_acceptance_effects_need_not_sum_to_an_identity_across_answers():
    protocol, strategy = sign_discrimination()
    cases = tuple(replace(c, after_first=(Continuation(0, (_scalar(F(1, 3)),)*2),))
                  for c in protocol.cases)
    report = replay(replace(protocol, cases=cases), strategy)
    assert report.acceptance_probability == F(1, 3)
    assert all(t.accepted_mass == t.probability/3 for t in report.transcripts)


def test_destructive_measurement_steers_the_verifier_reference_but_resets_local_memory():
    p, s = correlated_memory()
    projectors = (M(((1, 0), (0, 0))), M(((0, 0), (0, 1))))
    after = tuple(Continuation(y, (projectors[y], scale(identity(2), 0))) for y in range(2))
    p = replace(p, cases=(replace(p.cases[0], after_first=after),))
    # K_y = |0><y|; the Bell pair collapses to |y>_R |0>_memory.
    first = FirstMove(0, tuple((KrausTerm(1, M(((1-y, y), (0, 0)))),) for y in range(2)))
    final = tuple(FinalMove(LocalHistory(0, y, y), projectors) for y in range(2))
    report = replay(p, replace(s, first=(first,), final=final))
    assert [t.probability for t in report.transcripts] == [F(1, 2), 0, F(1, 2), 0]
    assert report.acceptance_probability == 1
