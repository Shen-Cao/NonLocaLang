"""Independent stabilizer law and wire-sensitive controls for the four-qubit fixture."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from nonlocalgames.protocols.interactive import (
    FinalMove, FirstMove, KrausTerm, LocalHistory, LocalStrategy, replay,
)
from nonlocalgames.protocols.interactive.pauli_frame import adaptive_pauli_frame
from nonlocalgames.quantum.exact import RationalMatrix
from nonlocalgames.quantum.exact.algebra import identity, scale


@pytest.mark.parametrize("stale,scoring,score", [
    (False, "frame", 1), (True, "frame", F(3, 4)),
    (False, "zero", F(1, 2)), (False, "swapped", F(1, 2)),
    (True, "zero", F(5, 8)), (True, "swapped", F(1, 2)),
])
def test_every_transcript_against_the_bell_stabilizers(stale, scoring, score):
    report = replay(*adaptive_pauli_frame(stale_right=stale, scoring=scoring))
    actual = {(t.private_case, t.history, t.final_answer): (t.probability, t.accepted_mass)
              for t in report.transcripts}
    assert len(report.transcripts) == len(actual) == 32
    for c, p, q, a, d in product(range(2), repeat=5):
        b = c ^ p
        parity = p if b == 0 else q  # endpoint ZZ and XX eigenvalue bits, respectively
        # Each (c,p,q) costs 1/8. In a common basis one of two equally likely
        # pairs occurs; crossed X/Z on a Bell state gives four equally likely pairs.
        if stale and p:
            mass = F(1, 32)
        else:
            mass = F(1, 16) if (a ^ d) == parity else 0
        target = {"frame": parity, "swapped": q if b == 0 else p, "zero": 0}[scoring]
        expected_win = mass if (a ^ d) == target else 0
        key = (c, LocalHistory(0, 2*p + q, b), 2*a + d)
        assert actual[key] == (mass, expected_win)
    assert report.acceptance_probability == score


def test_controls_change_only_the_named_device_or_scoring_rule():
    protocol, strategy = adaptive_pauli_frame()
    same_verifier, stale = adaptive_pauli_frame(stale_right=True)
    assert same_verifier == protocol and stale.first == strategy.first
    nominal_by_view = {m.history: m.effects for m in strategy.final}
    for move in stale.final:
        # Only histories with p=1 cross the bases.
        assert (move.effects == nominal_by_view[move.history]) == (move.history.first_answer < 2)
    for scoring in ("zero", "swapped"):
        changed, same_strategy = adaptive_pauli_frame(scoring=scoring)
        assert same_strategy == strategy
        for new, original in zip(changed.cases, protocol.cases, strict=True):
            assert replace(new, after_first=original.after_first) == original
            assert [c.question for c in new.after_first] == [
                c.question for c in original.after_first]


def test_tensor_factor_order_bell_sign_and_asymmetric_endpoint_labels():
    protocol, strategy = adaptive_pauli_frame(stale_right=True)
    assert (protocol.reference_dimension, protocol.message_dimension) == (1, 16)
    # Big-endian basis A,L,R,B: two Φ+ states have support |a a d d>.
    support = {0b0000, 0b0011, 0b1100, 0b1111}
    for case in protocol.cases:
        assert case.question == 0 and case.probability == F(1, 2)
        for i, j in product(range(16), repeat=2):
            assert case.state.real[i][j] == (F(1, 4) if i in support and j in support else 0)
            assert case.state.imag[i][j] == 0
    # At fixed A=B=0, the two inner basis vectors have parity p. Their
    # cross term is where the reported XX bit q is visible.
    first, = strategy.first
    assert first.question == 0 and len(first.outcomes) == 4
    for y, (term,) in enumerate(first.outcomes):
        p, q = divmod(y, 2)
        i, j = 2*p, 4 + 2*(1-p)
        k = term.operator.real
        assert term.weight == 1 and k[i][i] == k[j][j] == F(1, 2)
        assert k[i][j] == k[j][i] == F((-1)**q, 2)
        assert k[i ^ 2][i ^ 2] == k[i][i ^ 1] == k[i][i ^ 8] == 0

    final = {m.history: m.effects for m in strategy.final}
    # Symmetric Bell statistics cannot tell which endpoint is stale. For
    # first answer 10 and final answer 10, inspect both crossed POVMs directly.
    zx = final[LocalHistory(0, 2, 0)][2].real  # |1><1| on A; |+><+| on B
    assert zx[8][9] == F(1, 2) and zx[0][1] == 0
    xz = final[LocalHistory(0, 2, 1)][2].real  # |-><-| on A; |0><0| on B
    assert xz[0][8] == F(-1, 2) and xz[1][9] == 0


def test_phase_flip_on_one_source_reverses_only_x_parity():
    protocol, strategy = adaptive_pauli_frame()
    # Directly prepare Φ+ on AL and Φ− on RB. Every computational
    # population stays unchanged; a population-only simulator would miss it.
    signs = {0: 1, 3: -1, 12: 1, 15: -1}
    state = RationalMatrix(tuple(tuple(F(signs.get(i, 0)*signs.get(j, 0), 4)
                                       for j in range(16)) for i in range(16)))
    flipped = replace(protocol, cases=tuple(replace(c, state=state) for c in protocol.cases))
    report = replay(flipped, strategy)
    for t in report.transcripts:
        p, q = divmod(t.history.first_answer, 2)
        a, d = divmod(t.final_answer, 2)
        b = t.history.second_question
        physical = p if b == 0 else q ^ 1
        assert t.probability == (F(1, 16) if (a ^ d) == physical else 0)
        assert t.accepted_mass == (t.probability if b == 0 else 0)
    assert report.acceptance_probability == F(1, 2)


def test_a_fixed_classical_reply_also_wins_so_the_frame_game_is_not_a_self_test():
    protocol, _ = adaptive_pauli_frame()
    eye = identity(16)
    zero = scale(eye, 0)
    first = FirstMove(0, ((KrausTerm(1, eye),), (), (), ()))  # y=00 regardless of state
    final = tuple(FinalMove(LocalHistory(0, y, b), (eye, zero, zero, zero))
                  for y, b in product(range(4), range(2)))  # final answer 00
    report = replay(protocol, LocalStrategy((first,), final))
    assert report.acceptance_probability == 1
    assert all(t.history.first_answer == t.final_answer == 0 for t in report.transcripts
               if t.probability)


def test_control_names_and_flag_values_are_checked():
    with pytest.raises(TypeError, match="stale_right"):
        adaptive_pauli_frame(stale_right=1)
    with pytest.raises(ValueError, match="scoring"):
        adaptive_pauli_frame(scoring="stale")
