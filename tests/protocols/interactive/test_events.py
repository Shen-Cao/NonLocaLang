"""Check the event compiler at the public replay entry point."""
from dataclasses import replace
from enum import IntEnum
from fractions import Fraction as F
from importlib import import_module
from unittest.mock import ANY

import pytest
from numpy import bool_, int64

from nonlocalgames.protocols.interactive import LocalHistory, ReplayLimits, TwoRoundProgram, replay
from nonlocalgames.protocols.interactive.events import (
    ChooseContinuation, ClassicalRegister, Deliver, Instrument, Measure, Prepare, QuantumRegister,
    Reply, Score,
)
from nonlocalgames.protocols.interactive.examples import (
    correlated_memory, sign_discrimination, split_bell_label,
)
from nonlocalgames.protocols.interactive.pauli_frame import adaptive_pauli_frame
from nonlocalgames.quantum.exact import ExactBudgetExceeded
from nonlocalgames.quantum.exact.algebra import scale


def _change(program, i, **fields):
    return replace(program, events=(*program.events[:i], replace(program.events[i], **fields),
                                    *program.events[i+1:]))


@pytest.fixture
def program():
    return TwoRoundProgram.from_tables(*split_bell_label())


@pytest.mark.parametrize("memory, independent, expected", [
    ("retain", False, 1), ("dephase", False, F(1, 2)),
    ("reset", False, F(1, 4)), ("retain", True, F(1, 4)),
])
def test_split_message_all_sixteen_leaf_pairs(memory, independent, expected):
    protocol, strategy = split_bell_label(memory, independent=independent)
    program = TwoRoundProgram.from_tables(protocol, strategy)
    p, s = program.lower()
    assert (p, s) == (protocol, strategy)
    assert p.cases is protocol.cases and s.first is strategy.first and s.final is strategy.final
    got = replay(program)
    assert got == replay(protocol, strategy)
    assert len(got.transcripts) == 16 and got.acceptance_probability == expected
    for t in got.transcripts:
        label, z = t.private_case, t.final_answer
        assert t.history == LocalHistory(0, 0, 0)
        if independent or memory == "reset":
            prob = F(1, 16)
        elif memory == "dephase":
            prob = F(int(z // 2 == label // 2), 8)
        else:
            prob = F(int(z == label), 4)
        assert (t.probability, t.accepted_mass) == (prob, prob if z == label else 0)


@pytest.mark.parametrize("stale,scoring,expected", [
    (False, "frame", 1), (True, "frame", F(3, 4)), (True, "zero", F(5, 8)),
])
def test_adaptive_frame_all_thirty_two_leaf_pairs(stale, scoring, expected):
    p, s = adaptive_pauli_frame(stale_right=stale, scoring=scoring)
    got = replay(TwoRoundProgram.from_tables(p, s))
    assert got == replay(p, s)
    assert len(got.transcripts) == 32 and got.acceptance_probability == expected
    for t in got.transcripts:
        zz, xx = divmod(t.history.first_answer, 2)
        a, d = divmod(t.final_answer, 2)
        basis = t.private_case ^ zz
        assert t.history == LocalHistory(0, 2*zz + xx, basis)
        parity = (zz, xx)[basis]
        p_leaf = F(1, 32) if stale and zz else F(int((a ^ d) == parity), 16)
        win = (a ^ d) == (0 if scoring == "zero" else parity)
        assert (t.probability, t.accepted_mass) == (p_leaf, p_leaf if win else 0)


def test_reference_effect_changes_accepted_mass_without_changing_leaf_probability():
    # A non-scalar terminal reference measurement. Scale each correct X effect
    # by 2/3: the output distribution stays unchanged, but every accepted mass shrinks.
    p, s = correlated_memory()
    case, = p.cases
    follow = tuple(replace(cont, accept=tuple(scale(e, F(2, 3)) for e in cont.accept))
                   for cont in case.after_first)
    p = replace(p, cases=(replace(case, after_first=follow),))
    report = replay(TwoRoundProgram.from_tables(p, s))
    assert report == replay(p, s)
    assert report.acceptance_probability == F(2, 3)
    assert [(t.probability, t.accepted_mass) for t in report.transcripts] == [(F(1, 4), F(1, 6))]*4


def test_direct_authorship_renaming_and_commuting_deliveries_preserve_the_full_replay():
    p, s = split_bell_label("dephase")
    # Names are deliberately misleading: roles, not strings, determine locality.
    c, x, y, v, z = map(ClassicalRegister, ("sent", "private_case", "y", "v", "z"))
    r, later, early, mem = (QuantumRegister("early", 1), QuantumRegister("reference", 2),
                            QuantumRegister("late", 2), QuantumRegister("mem", 2))
    # Spelled from scratch, both wires move before their classical co-deliveries.
    events = (
        Prepare(c, x, r, later, early, p.cases), Deliver(early), Deliver(x),
        Instrument(x, early, y, mem, s.first), Reply(y), ChooseContinuation(c, y, v),
        Deliver(later), Deliver(v), Measure((x, y, v), (later, mem), z, s.final),
        Reply(z), Score(c, y, z, r),
    )
    assert replay(TwoRoundProgram(events)) == replay(p, s)
    assert TwoRoundProgram(events).lower() == (p, s)


def test_changed_operator_or_case_tables_at_the_corresponding_event_change_the_result(program):
    independent, _ = split_bell_label(independent=True)
    _, dephase = split_bell_label("dephase")
    assert replay(_change(program, 0, cases=independent.cases)).acceptance_probability == F(1, 4)
    assert replay(_change(program, 3, moves=dephase.first)).acceptance_probability == F(1, 2)
    move, = dephase.final
    shifted = replace(move, effects=(*move.effects[1:], move.effects[0]))
    assert replay(_change(program, 8, moves=(shifted,))).acceptance_probability == 0


def test_delivery_is_once_per_wire_and_never_discloses_the_private_case_or_reference(program):
    p, _, _, ins, _, cont, _, _, m, _, _ = program.events
    for slot, wrong in ((1, p.private_case), (2, p.reference), (2, p.second), (2, p.question),
                        (6, p.private_case), (7, p.reference), (7, p.first), (7, ins.memory),
                        (7, cont.question), (1, ins.answer), (6, m.answer), (6, ANY)):
        bad = _change(program, slot, register=wrong)
        with pytest.raises(ValueError, match="deliveries must"):
            replay(bad)


def test_local_reads_respect_timing_visible_classical_keys_and_ordered_quantum_cut(program):
    p, _, _, ins, _, cont, _, _, m, _, _ = program.events
    for wrong in (p.private_case, ins.answer, cont.question, m.answer, p.first):
        with pytest.raises(ValueError, match="First local instrument"):
            _change(program, 3, question=wrong).lower()
    for wrong in (p.second, p.reference, ins.memory, p.question):
        with pytest.raises(ValueError, match="First local instrument"):
            _change(program, 3, message=wrong).lower()
    for wrong in ((p.private_case, ins.answer, cont.question), m.history[::-1],
                  (*m.history, p.private_case), (*m.history[:2], m.answer)):
        with pytest.raises(ValueError, match="Final local history"):
            _change(program, 8, history=wrong).lower()
    for wrong in ((p.reference, ins.memory), (p.first, p.second), m.wires[::-1], (p.second,)):
        with pytest.raises(ValueError, match="Final local POVM"):
            _change(program, 8, wires=wrong).lower()
    # Sharing the same dimensions is no permission to read or transfer another wire.
    assert p.first.dimension == p.second.dimension == ins.memory.dimension


def test_verifier_may_select_continuation_and_score_only_after_the_correct_replies(program):
    p, _, _, ins, _, _, _, _, m, _, _ = program.events
    for slot, changes, message in (
        (4, {"answer": m.answer}, "First reply"),
        (5, {"first_answer": m.answer}, "Verifier continuation"),
        (5, {"private_case": p.question}, "Verifier continuation"),
        (9, {"answer": ins.answer}, "Final reply"),
        (10, {"reference": p.second}, "Verifier score"),
        (10, {"first_answer": m.answer}, "Verifier score"),
        (10, {"final_answer": p.question}, "Verifier score"),
    ):
        with pytest.raises(ValueError, match=message):
            replay(_change(program, slot, **changes))


def test_stage_order_is_checked_even_with_a_trivial_second_wire():
    program = TwoRoundProgram.from_tables(*sign_discrimination())
    ev = program.events
    assert ev[0].second.dimension == 1
    for i in range(len(ev)-1):
        swapped = TwoRoundProgram((*ev[:i], ev[i+1], ev[i], *ev[i+2:]))
        if i in (1, 6):
            assert swapped.lower() == program.lower()
        else:
            with pytest.raises(ValueError, match="Event"):
                swapped.lower()
    for invalid in (ev[:-1], ev[:7] + ev[8:], ev[:2] + ev[1:]):
        with pytest.raises(ValueError, match="exactly 11 events"):
            TwoRoundProgram(invalid).lower()


def test_bindings_cannot_alias_across_time_kinds_or_dimensions(program):
    p, _, _, ins, _, _, _, _, _, _, _ = program.events
    for slot, changes in (
        (0, {"question": p.private_case}), (3, {"memory": p.first}),
        (5, {"question": ins.answer}), (8, {"answer": p.question}),
        (0, {"first": QuantumRegister(p.reference.name, 3)}),
        (3, {"answer": ClassicalRegister(p.reference.name)}),
    ):
        with pytest.raises(ValueError, match="fresh"):
            _change(program, slot, **changes).lower()
    with pytest.raises(ValueError, match="first message dimension"):
        _change(program, 3, memory=replace(ins.memory, dimension=3)).lower()
    with pytest.raises(ValueError, match="First local instrument"):
        _change(program, 3, message=replace(p.first, dimension=3)).lower()
    with pytest.raises(TypeError, match="declared kind"):
        _change(program, 0, first=p.question).lower()


def test_wrong_containers_and_inexact_register_declarations_are_not_admitted(program):
    ev = program.events
    with pytest.raises(TypeError, match="events must be a tuple"):
        TwoRoundProgram(iter(ev)).lower()
    for slot, field in ((0, "cases"), (3, "moves"), (8, "moves"), (8, "wires"), (8, "history")):
        with pytest.raises(TypeError, match="tuple"):
            _change(program, slot, **{field: list(getattr(ev[slot], field))}).lower()
    for name in (1, "", "not a name", "X"*65, "χ"):
        with pytest.raises(ValueError, match="Register names"):
            ClassicalRegister(name)
        with pytest.raises(ValueError, match="Register names"):
            QuantumRegister(name, 1)
    for dimension in (0, -1, True, 1.0):
        with pytest.raises(ValueError, match="positive integer"):
            QuantumRegister("wire", dimension)
    with pytest.raises(TypeError, match="TwoRoundProtocol and LocalStrategy"):
        TwoRoundProgram.from_tables(program, ())


def test_table_adapter_canonicalizes_legacy_integral_dimensions_without_changing_replay():
    class Dim(IntEnum):
        ONE = 1
        TWO = 2

    p, s = split_bell_label()
    for ref, early, late in ((int64(1), Dim.TWO, int64(2)), (Dim.ONE, int64(2), Dim.TWO)):
        legacy = replace(p, reference_dimension=ref, message_dimension=early,
                         second_message_dimension=late)
        prog = TwoRoundProgram.from_tables(legacy, s)
        prep = prog.events[0]
        assert all(type(q.dimension) is int for q in (prep.reference, prep.first, prep.second))
        assert prog.lower() == (p, s)
        assert replay(prog) == replay(legacy, s)
    with pytest.raises(ValueError, match="positive integer"):
        QuantumRegister("direct", int64(2))


@pytest.mark.parametrize("dimension", [True, bool_(True), 1.0, F(1)])
def test_table_adapter_rejects_nonintegral_dimension_types_just_like_direct_replay(dimension):
    p, s = split_bell_label()
    for field in ("reference_dimension", "message_dimension", "second_message_dimension"):
        bad = replace(p, **{field: dimension})
        with pytest.raises(ValueError, match="Matrix dimensions must be positive integers"):
            replay(bad, s)
        with pytest.raises(ValueError, match="Matrix dimensions must be positive integers"):
            TwoRoundProgram.from_tables(bad, s)


def test_table_adapter_does_not_enforce_default_joint_dimension_budget():
    p, s = split_bell_label()
    too_large = replace(p, message_dimension=int64(17))
    prog = TwoRoundProgram.from_tables(too_large, s)
    assert prog.lower()[0].message_dimension == 17
    with pytest.raises(ExactBudgetExceeded, match="max_joint_dimension"):
        replay(prog)


def test_structural_admission_is_not_strategy_or_physics_validation(program):
    _, s = split_bell_label()
    with pytest.raises(TypeError, match="already supplies"):
        replay(program, s)
    move, = s.final
    # Only one local history exists for four private cases; no table row can
    # dispatch on a purported question not actually sent.
    bad = _change(program, 8, moves=(move, replace(move, history=LocalHistory(0, 0, 1))))
    assert bad.lower()[1].final == bad.events[8].moves
    with pytest.raises(ValueError, match="unused moves"):
        replay(bad)
    # Repeating a valid effect does not make a POVM.
    bad = _change(program, 8, moves=(replace(move, effects=(move.effects[0],)*4),))
    with pytest.raises(ValueError, match="sum to identity"):
        replay(bad)


def test_event_syntax_does_not_use_the_leaf_budget_or_downgrade_any_replay_budget(monkeypatch):
    defaults = ReplayLimits()
    small = replace(defaults, exact=replace(defaults.exact, max_events=4, max_matrix_entries=24))
    prog = TwoRoundProgram.from_tables(*sign_discrimination())
    assert len(prog.events) == 11 and replay(prog, limits=small).acceptance_probability == F(9, 10)
    p, s = split_bell_label()
    prog = TwoRoundProgram.from_tables(p, s)
    exact = replace(defaults.exact, max_joint_dimension=4, max_psd_size=8,
                    max_matrix_entries=148, max_events=16)
    limits = ReplayLimits(exact=exact, max_kraus_terms=1, max_dense_work=6308)
    assert replay(prog, limits=limits) == replay(p, s, limits=limits)
    dephase = TwoRoundProgram.from_tables(*split_bell_label("dephase"))

    module = import_module("nonlocalgames.protocols.interactive.replay")
    monkeypatch.setattr(module, "identity", lambda *a, **kw: pytest.fail("late size check"))
    for field, lower, msg in (("max_joint_dimension", 3, "dimension"),
                              ("max_psd_size", 7, "max_psd_size"),
                              ("max_matrix_entries", 147, "matrix entries"),
                              ("max_events", 15, "transcript leaves"),
                              ("max_scalar_bits", 1, "max_scalar_bits")):
        with pytest.raises(ExactBudgetExceeded, match=msg):
            replay(prog, limits=replace(limits, exact=replace(exact, **{field: lower})))
    with pytest.raises(ExactBudgetExceeded, match="max_dense_work"):
        replay(prog, limits=replace(limits, max_dense_work=6307))
    with pytest.raises(ExactBudgetExceeded, match="Kraus terms"):
        replay(dephase, limits=replace(defaults, max_kraus_terms=1))
