"""Malformed physics/shape data and pre-algebra budget controls."""
from dataclasses import replace
from fractions import Fraction as F
from importlib import import_module

import pytest

from nonlocalgames.protocols.interactive import (
    Continuation, FinalMove, FirstMove, KrausTerm, LocalHistory, ReplayLimits, replay,
)
from nonlocalgames.protocols.interactive.examples import correlated_memory, sign_discrimination
from nonlocalgames.quantum.exact import ExactBudget, ExactBudgetExceeded, RationalMatrix as M
from nonlocalgames.quantum.exact.algebra import identity, scale


def _with_case(protocol, index=0, **changes):
    cases = list(protocol.cases)
    cases[index] = replace(cases[index], **changes)
    return replace(protocol, cases=tuple(cases))


@pytest.mark.parametrize("value, error", [
    (F(-1, 2), ValueError), (F(3, 2), ValueError), (0.5, TypeError), (True, TypeError),
])
def test_prior_must_be_an_exact_probability(value, error):
    p, s = sign_discrimination()
    with pytest.raises(error):
        replay(_with_case(p, probability=value), s)


def test_complete_prior_and_all_states_checked_including_a_zero_prior():
    p, s = sign_discrimination()
    with pytest.raises(ValueError, match="sum to one"):
        replay(_with_case(p, probability=0), s)
    p = _with_case(_with_case(p, probability=0), 1, probability=1)
    assert sum(t.probability for t in replay(p, s).transcripts if t.private_case == 0) == 0
    for state, message in [(M(((2, 0), (0, -1))), "not PSD"), (identity(2), "trace one"),
                           (M(((1, 1), (0, 0))), "Hermitian")]:
        with pytest.raises(ValueError, match=message):
            replay(_with_case(p, state=state), s)


@pytest.mark.parametrize("weight, op, message", [
    (F(-1), identity(2), "nonnegative"), (F(2), identity(2), "trace preserving"),
    (F(1), identity(1), "dimension"),
])
def test_instrument_checked_even_with_a_correct_terminal_povm(weight, op, message):
    p, s = sign_discrimination()
    with pytest.raises(ValueError, match=message):
        replay(p, replace(s, first=(FirstMove(0, ((KrausTerm(weight, op),),)),)))


def test_final_measurements_require_psd_completeness_and_verifier_requires_both_effect_bounds():
    p, s = sign_discrimination()
    iq = identity(2)
    for effects, message in [((iq, iq), "sum to identity"),
                             ((scale(iq, -1), scale(iq, 2)), "not PSD")]:
        with pytest.raises(ValueError, match=message):
            replay(p, replace(s, final=(replace(s.final[0], effects=effects),)))
    for x in (-1, 2):
        tests = (M(((x,),)), M(((0,),)))
        with pytest.raises(ValueError, match="not PSD"):
            replay(_with_case(p, after_first=(Continuation(0, tests),)), s)


def test_invalid_quantum_operation_on_an_impossible_first_outcome_is_not_ignored():
    p, s = correlated_memory()
    two = scale(identity(2), 2)
    first = (FirstMove(0, ((KrausTerm(1, identity(2)),), ())),)
    final = (s.final[0], replace(s.final[1], effects=(two, two)))
    with pytest.raises(ValueError, match="sum to identity"):
        replay(p, replace(s, first=first, final=final))


def test_canonical_local_tables_no_hidden_case_or_unreceived_question_can_index_a_move():
    p, s = sign_discrimination()
    _, expose = sign_discrimination(reveal=True)
    with pytest.raises(ValueError, match="unused moves"):
        replay(p, expose)  # a second row selected by the private bit is not allowed
    with pytest.raises(ValueError, match="Duplicate first question"):
        replay(p, replace(s, first=s.first*2))
    with pytest.raises(ValueError, match="Duplicate final history"):
        replay(p, replace(s, final=s.final*2))
    for bad in (LocalHistory(1, 0, 0), LocalHistory(0, 1, 0), LocalHistory(0, 0, 1)):
        with pytest.raises(ValueError, match="Missing final move"):
            replay(p, replace(s, final=(replace(s.final[0], history=bad),)))
    with pytest.raises(ValueError, match="Missing first move"):
        replay(_with_case(p, question=1), s)


def test_inconsistent_answer_alphabets_and_empty_tables_are_rejected():
    p, s = sign_discrimination()
    c = p.cases[0].after_first[0]
    with pytest.raises(ValueError, match="first-answer alphabet"):
        replay(_with_case(p, after_first=(c, c)), s)
    with pytest.raises(ValueError, match="final-answer alphabet"):
        replay(_with_case(p, after_first=(replace(c, accept=(identity(1),)),)), s)
    with pytest.raises(ValueError, match="cannot be empty"):
        replay(replace(p, cases=()), s)
    with pytest.raises(ValueError, match="cannot be empty"):
        replay(p, replace(s, final=(FinalMove(s.final[0].history, ()),)))


@pytest.mark.parametrize("bad", [-1, True, 2**31, 0.0, "private", (0, 1)])
def test_question_is_a_bounded_classical_label_not_a_metadata_or_future_state_channel(bad):
    p, s = sign_discrimination()
    with pytest.raises(ValueError, match="Classical labels"):
        replay(_with_case(p, question=bad), s)


def test_strategies_are_declarative_and_require_materialized_tables():
    p, s = sign_discrimination()
    with pytest.raises(TypeError, match="LocalStrategy"):
        replay(p, lambda private_state: s)
    with pytest.raises(TypeError, match="tuple"):
        replay(p, replace(s, first=iter(s.first)))
    with pytest.raises(TypeError, match="RationalMatrix"):
        replay(_with_case(p, state=[[1, 0], [0, 0]]), s)
    with pytest.raises(ValueError, match="matrix dimensions|Matrix dimensions"):
        replay(replace(p, message_dimension=True), s)


@pytest.mark.parametrize("pair,limits,message", [
    (sign_discrimination, ReplayLimits(exact=ExactBudget(max_events=3)), "transcript leaves"),
    (sign_discrimination, ReplayLimits(exact=ExactBudget(max_matrix_entries=23)), "total matrix"),
    (sign_discrimination, ReplayLimits(exact=ExactBudget(max_scalar_bits=2)), "max_scalar_bits"),
    (correlated_memory, ReplayLimits(exact=ExactBudget(max_joint_dimension=3)), "dimension"),
    (correlated_memory, ReplayLimits(exact=ExactBudget(max_psd_size=7)), "max_psd_size"),
    (correlated_memory, ReplayLimits(max_kraus_terms=1), "max_kraus_terms"),
    (correlated_memory, ReplayLimits(max_dense_work=1), "max_dense_work"),
])
def test_small_budgets_fail_during_preflight_before_dense_physics_checks(monkeypatch, pair,
                                                                          limits, message):
    # Deliberately build fixtures first: they are trusted Python inputs, not file decoding.
    p, s = pair()
    module = import_module("nonlocalgames.protocols.interactive.replay")
    monkeypatch.setattr(module, "_check_physics", lambda *args: pytest.fail("late budget check"))
    with pytest.raises(ExactBudgetExceeded, match=message):
        replay(p, s, limits=limits)


def test_inclusive_size_limits_and_guard_for_repeated_empty_outcome_groups():
    p, s = sign_discrimination()
    limits = ReplayLimits(exact=ExactBudget(max_matrix_entries=24, max_events=4))
    assert replay(p, s, limits=limits).acceptance_probability == F(9, 10)
    with pytest.raises(ExactBudgetExceeded, match="instrument outcomes"):
        replay(p, replace(s, first=(FirstMove(0, ((),)*5),)), limits=limits)


def test_limits_must_be_positive_integers():
    with pytest.raises(TypeError, match="ExactBudget"):
        ReplayLimits(exact=1)
    for kw in ({"max_kraus_terms": True}, {"max_dense_work": 0}, {"max_dense_work": 3.5}):
        with pytest.raises(ValueError, match="positive integer"):
            ReplayLimits(**kw)


def test_exact_derived_values_can_exceed_stored_scalar_budget_without_rounding():
    p, s = sign_discrimination()
    # The entries of the supplied ρ± fit in 3 bits, but 9/20 and 1/20 do not.
    with pytest.raises(ExactBudgetExceeded, match="max_scalar_bits"):
        replay(p, s, limits=ReplayLimits(exact=ExactBudget(max_scalar_bits=3)))
