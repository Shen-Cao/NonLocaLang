"""Bound declarative tables and compile local views before dense verification."""
from fractions import Fraction

from ...quantum.exact import ExactBudgetExceeded, RationalMatrix
from ...quantum.exact.budgets import bounded_dimension
from ...quantum.exact.scalars import exact_scalar
from .model import (
    BobMove, Continuation, FinalMove, FirstMove, KrausTerm, LocalHistory, LocalStrategy,
    PrivateCase, ReplayLimits, TwoProverProtocol, TwoProverStrategy, TwoRoundProtocol,
)


def _bounded(n, maximum, name):
    if n > maximum:
        raise ExactBudgetExceeded(f"Interactive replay exceeds {name}")


def _table(values, maximum, name, *, allow_empty=False):
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    _bounded(len(values), maximum, name)
    if not values and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    return values


def _label(value):
    if type(value) is not int or value < 0 or value.bit_length() > 31:
        raise ValueError("Classical labels must be nonnegative 31-bit integers")
    return value


class _Meter:
    def __init__(self, limits):
        self.limits = limits
        self.entries = self.work = 0

    def matrix(self, matrix, dim):
        if not isinstance(matrix, RationalMatrix):
            raise TypeError("Interactive data requires RationalMatrix objects")
        if matrix.dimension != dim:
            raise ValueError("Matrix dimension differs from its register")
        matrix.check_budget(self.limits.exact)
        self.entries += dim**2
        _bounded(self.entries, self.limits.exact.max_matrix_entries, "total matrix entries")

    def charge(self, units):
        self.work += units
        _bounded(self.work, self.limits.max_dense_work, "max_dense_work")


def prepare(protocol, strategy, limits):
    if not isinstance(protocol, TwoRoundProtocol) or not isinstance(strategy, LocalStrategy):
        raise TypeError("Expected TwoRoundProtocol and LocalStrategy")
    return _prepare(protocol.reference_dimension, protocol.second_message_dimension,
                    protocol.message_dimension, protocol.cases, strategy.first, strategy.final,
                    limits, separate_provers=False)


def prepare_two_prover(protocol, strategy, limits):
    if not isinstance(protocol, TwoProverProtocol) or not isinstance(strategy, TwoProverStrategy):
        raise TypeError("Expected TwoProverProtocol and TwoProverStrategy")
    return _prepare(protocol.reference_dimension, protocol.bob_dimension, protocol.alice_dimension,
                    protocol.cases, strategy.alice, strategy.bob, limits, separate_provers=True)


def _prepare(r, q2, q1, private_cases, first_moves, final_moves, limits, *, separate_provers):
    if not isinstance(limits, ReplayLimits):
        raise TypeError("Expected ReplayLimits")
    b = limits.exact
    if separate_provers:
        _bounded(2, b.max_parties, "max_parties")
    r = bounded_dimension(r, b)
    q1 = bounded_dimension(q1, b)
    q2 = bounded_dimension(q2, b)
    local = q2 if separate_provers else q2*q1  # Bob owns only Q_B, never Alice's register.
    d = bounded_dimension(r*q2*q1, b)
    # The preparation is the largest PSD input; reference and final effects fit too.
    _bounded(2*d, b.max_psd_size, "max_psd_size")
    meter = _Meter(limits)
    first, final, cases = {}, {}, []
    groups = nterms = nfinal = 0
    for move in _table(first_moves, b.max_events, "first moves"):
        if not isinstance(move, FirstMove):
            raise TypeError("Expected FirstMove")
        x = _label(move.question)
        if x in first:
            raise ValueError("Duplicate first question in local strategy")
        outcomes = _table(move.outcomes, b.max_events, "instrument outcomes")
        groups += len(outcomes)
        _bounded(groups, b.max_events, "instrument outcomes")
        if separate_provers:
            meter.charge(len(outcomes)*q1*q1)  # aggregate the saved Alice outcome effects
        resolved = []
        for group in outcomes:
            terms = []
            for term in _table(group, limits.max_kraus_terms, "Kraus terms", allow_empty=True):
                if not isinstance(term, KrausTerm):
                    raise TypeError("Expected KrausTerm")
                nterms += 1
                _bounded(nterms, limits.max_kraus_terms, "max_kraus_terms")
                w = exact_scalar(term.weight, max_bits=b.max_scalar_bits)
                if w < 0:
                    raise ValueError("Kraus weights must be nonnegative")
                meter.matrix(term.operator, q1)
                meter.charge(q1**3 + 3*q1*q1)  # K†K and accumulate completeness
                terms.append((w, term.operator))
            resolved.append(tuple(terms))
        first[x] = tuple(resolved)
    for move in _table(final_moves, b.max_events, "final moves"):
        if separate_provers:
            if not isinstance(move, BobMove):
                raise TypeError("Expected BobMove with only Bob's question")
            key = _label(move.question)
            if key in final:
                raise ValueError("Duplicate Bob question in local strategy")
        else:
            if not isinstance(move, FinalMove) or not isinstance(move.history, LocalHistory):
                raise TypeError("Expected FinalMove with LocalHistory")
            key = move.history
            for label in (key.first_question, key.first_answer, key.second_question):
                _label(label)
            if key in final:
                raise ValueError("Duplicate final history in local strategy")
        effects = _table(move.effects, b.max_events, "final POVM")
        nfinal += len(effects)
        _bounded(nfinal, b.max_events, "final POVM outcomes")
        for effect in effects:
            meter.matrix(effect, local)
            meter.charge((2*local)**3 + local*local)  # realified PSD and completeness
        final[key] = effects

    used_first, used_final = set(), set()
    leaves = 0
    mass = Fraction(0)
    for case in _table(private_cases, b.max_events, "private cases"):
        if not isinstance(case, PrivateCase):
            raise TypeError("Expected PrivateCase")
        p = exact_scalar(case.probability, max_bits=b.max_scalar_bits)
        if not 0 <= p <= 1:
            raise ValueError("Case prior must lie in [0, 1]")
        mass = exact_scalar(mass + p, max_bits=b.max_scalar_bits)
        meter.matrix(case.state, d)
        meter.charge((2*d)**3)
        x = _label(case.question)
        follow = _table(case.after_first, b.max_events, "verifier continuations")
        if x not in first or len(follow) != len(first[x]):
            raise ValueError("Missing first move or inconsistent first-answer alphabet")
        used_first.add(x)
        for y, cont in enumerate(follow):
            if not isinstance(cont, Continuation):
                raise TypeError("Expected Continuation")
            v = _label(cont.question)
            key = v if separate_provers else LocalHistory(x, y, v)
            tests = _table(cont.accept, b.max_events, "verifier acceptance effects")
            if key not in final or len(tests) != len(final[key]):
                raise ValueError("Missing final move or inconsistent final-answer alphabet")
            used_final.add(key)
            leaves += len(tests)
            _bounded(leaves, b.max_events, "transcript leaves")
            if separate_provers:
                meter.charge(len(tests)*(q2*q1)**2)  # B[v,z] ⊗ A[x,y], once per leaf
            else:
                meter.charge(len(first[x][y])*(2*d**3 + 4*d*d) + d*d)
            for test in tests:
                meter.matrix(test, r)
                meter.charge(2*(2*r)**3 + 4*d*d + r*r)  # 0 <= E <= I; two Born traces
        cases.append((p, case))
    if mass != 1:
        raise ValueError("Private case probabilities must sum to one")
    if set(first) != used_first or set(final) != used_final:
        raise ValueError("Strategy must have exactly the declared local views; found unused moves")
    return (r, q2, q1), cases, first, final
