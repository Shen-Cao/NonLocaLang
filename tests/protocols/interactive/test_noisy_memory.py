"""Closed-form leaf laws for three different measurements on the same split-Bell states."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest
import sympy as sp

from nonlocalgames.protocols.interactive import FirstMove, KrausTerm, LocalHistory, replay
from nonlocalgames.protocols.interactive.examples import split_bell_label
from nonlocalgames.protocols.interactive.noisy_memory import noisy_split_bell_label
from nonlocalgames.quantum.exact import RationalMatrix
from nonlocalgames.quantum.exact.algebra import identity, kron, scale, subtract


def _check_leaves(report, joint, wanted):
    assert {t.history for t in report.transcripts} == {LocalHistory(0, 0, 0)}
    actual = {(t.private_case, t.final_answer): (t.probability, t.accepted_mass)
              for t in report.transcripts}
    assert len(report.transcripts) == len(joint)
    assert actual == {(g, z): (p, p if z == wanted[g] else 0) for (g, z), p in joint.items()}


@pytest.mark.parametrize("channel, relative", [
    ("damping", (81, 16, 36, 36)), ("pauli_twirl", (81, 16, 36, 36)),
    ("phase_plus", (68, 29, 36, 36)), ("phase_minus", (68, 29, 36, 36)),
])
def test_bell_readout_all_sixteen_cells_not_just_diagonal_success(channel, relative):
    report = replay(*noisy_split_bell_label(channel))
    # XOR order for Φ+, Φ−, Ψ+, Ψ− is I,Z,X,Y. Each g also contributes its 1/4 prior.
    joint = {(g, z): F(relative[g ^ z], 4*169) for g, z in product(range(4), repeat=2)}
    _check_leaves(report, joint, range(4))
    assert report.acceptance_probability == F(relative[0], 169)


@pytest.mark.parametrize("channel, plus", [
    ("damping", F(313, 338)), ("pauli_twirl", F(1, 2)),
    ("phase_plus", F(313, 338)), ("phase_minus", F(313, 338)),
])
def test_z_on_stored_wire_reads_the_translation_of_its_maximally_mixed_input(channel, plus):
    report = replay(*noisy_split_bell_label(channel, readout="memory_z"))
    joint = {(g, z): (plus, 1-plus)[z]/4 for g, z in product(range(4), range(2))}
    _check_leaves(report, joint, (0, 0, 0, 0))
    assert report.acceptance_probability == plus


@pytest.mark.parametrize("channel, imag", [
    ("damping", F(0)), ("pauli_twirl", F(0)), ("phase_plus", F(4, 13)), ("phase_minus", F(-4, 13)),
])
def test_xy_parity_resolves_private_rows_but_not_pooled_readout(channel, imag):
    report = replay(*noisy_split_bell_label(channel, readout="xy_parity"))
    # In Q2 ⊗ M order, Y⊗X sends |0,b> to i|1,1-b> and |1,b> to -i|0,1-b>.
    # A jump contributes no cross term; the no-jump Bell ket gives sign(g) Im(a).
    chi = (1, -1, -1, 1)
    joint = {(g, z): F(1, 8)*(1 + (-1)**z * chi[g] * imag)
             for g, z in product(range(4), range(2))}
    _check_leaves(report, joint, (0, 1, 1, 0))
    assert report.acceptance_probability == (1+imag)/2
    assert sum(t.probability for t in report.transcripts if t.final_answer == 0) == F(1, 2)


def test_changes_are_confined_to_the_named_operations():
    ideal, retained = split_bell_label()
    for readout in ("bell", "memory_z", "xy_parity"):
        verifier, reference = noisy_split_bell_label("damping", readout=readout)
        assert all(replace(c, after_first=old.after_first) == old
                   for c, old in zip(verifier.cases, ideal.cases, strict=True))
        assert len(reference.final) == 1 and reference.final[0].history == LocalHistory(0, 0, 0)
        if readout == "bell":
            assert verifier == ideal and reference.final == retained.final
        for channel in ("damping", "pauli_twirl", "phase_plus", "phase_minus"):
            p, s = noisy_split_bell_label(channel, readout=readout)
            _, bell = noisy_split_bell_label(channel)
            first, = s.first
            assert first.question == 0 and len(first.outcomes) == 1
            assert p == verifier and s.final == reference.final and s.first == bell.first


def test_declared_first_operations_as_maps_on_a_general_two_by_two_matrix():
    # Independent symbolic check; matching just the three recorded measurements is weaker.
    rho = sp.Matrix(2, 2, sp.symbols("u:4"))

    def damp(r, a):
        return sp.Matrix([[r[0, 0] + sp.Rational(144, 169)*r[1, 1], sp.conjugate(a)*r[0, 1]],
                          [a*r[1, 0], sp.Rational(25, 169)*r[1, 1]]])

    real = sp.Rational(5, 13)
    x, z = sp.Matrix([[0, 1], [1, 0]]), sp.diag(1, -1)
    paulis = (sp.eye(2), x, z, sp.I*x*z)
    twirled = sum((p*damp(p.H*rho*p, real)*p.H for p in paulis), sp.zeros(2))/4
    for channel, expected in [
        ("damping", damp(rho, real)), ("pauli_twirl", twirled),
        ("phase_plus", damp(rho, (3+4*sp.I)/13)), ("phase_minus", damp(rho, (3-4*sp.I)/13)),
    ]:
        _, strategy = noisy_split_bell_label(channel)
        terms, = strategy.first[0].outcomes
        actual = sp.zeros(2)
        for term in terms:
            k = sp.Matrix(term.operator.real) + sp.I*sp.Matrix(term.operator.imag)
            actual += term.weight*k*rho*k.H
        assert (actual - expected).expand() == sp.zeros(2)


def test_a_classical_record_can_win_the_parity_verifier_with_different_measurements():
    protocol, template = noisy_split_bell_label(readout="xy_parity")
    # Measure Y on Q1 and store ± as a classical 0/1 in M; still return just the common ack.
    plus = RationalMatrix(((1, 0), (0, 0)), ((0, -1), (0, 0)))
    minus = RationalMatrix(((0, 0), (1, 0)), ((0, 0), (0, 1)))
    ack = FirstMove(0, (tuple(KrausTerm(F(1, 2), k) for k in (plus, minus)),))
    # χ(g) is the eigenvalue of -Y⊗Y on the original Bell state. Read Y on Q2
    # and negate its product with the recorded bit's sign, Z_M.
    y = RationalMatrix(((0, 0), (0, 0)), ((0, -1), (1, 0)))
    z = RationalMatrix(((1, 0), (0, -1)))
    eye = identity(4)
    plus_effect = scale(subtract(eye, kron(y, z)), F(1, 2))
    final = replace(template.final[0], effects=(plus_effect, subtract(eye, plus_effect)))
    report = replay(protocol, replace(template, first=(ack,), final=(final,)))
    wanted = (0, 1, 1, 0)
    _check_leaves(report, {(g, z): F(z == wanted[g], 4)
                           for g, z in product(range(4), range(2))}, wanted)
    assert report.acceptance_probability == 1


def test_unknown_modes_fail_without_falling_back_to_a_different_physical_model():
    with pytest.raises(ValueError, match="channel"):
        noisy_split_bell_label("not_a_channel")
    with pytest.raises(ValueError, match="readout"):
        noisy_split_bell_label(readout="not_a_readout")
