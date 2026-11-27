"""A closed event language for the one-prover, two-round replay dialect.

Register names are local binders, not dispatch keys. Only a checked event order
and the existing declarative tables lower; there are no executable event hooks.
"""
from dataclasses import dataclass
from numbers import Integral
from operator import index

from .model import FinalMove, FirstMove, LocalStrategy, PrivateCase, TwoRoundProtocol


def _check_name(name):
    if (type(name) is not str or not 1 <= len(name) <= 64
            or not name.isascii() or not name.isidentifier()):
        raise ValueError("Register names must be ASCII identifiers of at most 64 characters")


def _table_dimension(value) -> int:
    # Match bounded_dimension's type admission without imposing a replay limit here.
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("Matrix dimensions must be positive integers")
    return index(value)


@dataclass(frozen=True, slots=True)
class ClassicalRegister:
    """A symbolic slot for a classical label, not the value of that label."""
    name: str

    def __post_init__(self):
        _check_name(self.name)


@dataclass(frozen=True, slots=True)
class QuantumRegister:
    """A symbolic wire with a fixed positive Hilbert-space dimension."""
    name: str
    dimension: int

    def __post_init__(self):
        _check_name(self.name)
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("Quantum register dimension must be a positive integer")


@dataclass(frozen=True, slots=True)
class Prepare:
    """Verifier privately draws a case and prepares reference ⊗ second ⊗ first."""
    private_case: ClassicalRegister
    question: ClassicalRegister
    reference: QuantumRegister
    second: QuantumRegister
    first: QuantumRegister
    cases: tuple[PrivateCase, ...]


@dataclass(frozen=True, slots=True)
class Deliver:
    """Verifier to prover: disclose a classical question or transfer a quantum wire."""
    register: ClassicalRegister | QuantumRegister


@dataclass(frozen=True, slots=True)
class Instrument:
    """Read the delivered question and consume Q1; bind a fresh answer and memory."""
    question: ClassicalRegister
    message: QuantumRegister
    answer: ClassicalRegister
    memory: QuantumRegister
    moves: tuple[FirstMove, ...]


@dataclass(frozen=True, slots=True)
class Reply:
    """Prover to verifier: disclose this round's classical answer, keeping a copy."""
    answer: ClassicalRegister


@dataclass(frozen=True, slots=True)
class ChooseContinuation:
    """Bind the question in cases[private_case].after_first[first_answer]."""
    private_case: ClassicalRegister
    first_answer: ClassicalRegister
    question: ClassicalRegister


@dataclass(frozen=True, slots=True)
class Measure:
    """Apply this history's POVM to the ordered local wires and bind the final answer."""
    history: tuple[ClassicalRegister, ClassicalRegister, ClassicalRegister]
    wires: tuple[QuantumRegister, QuantumRegister]
    answer: ClassicalRegister
    moves: tuple[FinalMove, ...]


@dataclass(frozen=True, slots=True)
class Score:
    """Select the terminal effect by the verifier's case and both received replies."""
    private_case: ClassicalRegister
    first_answer: ClassicalRegister
    final_answer: ClassicalRegister
    reference: QuantumRegister


Event = Prepare | Deliver | Instrument | Reply | ChooseContinuation | Measure | Score
_ORDER = (Prepare, Deliver, Deliver, Instrument, Reply, ChooseContinuation,
          Deliver, Deliver, Measure, Reply, Score)


def _matches(operands, expected):
    # Type first: equality of arbitrary Python objects is not register identity.
    return len(operands) == len(expected) and all(
        type(actual) is type(bound) and actual == bound for actual, bound in zip(operands, expected)
    )


def _deliveries(left, right, question, wire, stage):
    pair = (left.register, right.register)
    if not (_matches(pair, (question, wire)) or _matches(pair, (wire, question))):
        raise ValueError(f"{stage} deliveries must disclose {question.name} and transfer "
                         f"{wire.name}, each exactly once")


def _reads(actual, expected, stage):
    if type(actual) is not tuple:
        raise TypeError(f"{stage} register operands must be a tuple")
    if not _matches(actual, expected):
        raise ValueError(f"{stage} must read exactly {', '.join(r.name for r in expected)} in order")


@dataclass(frozen=True, slots=True)
class TwoRoundProgram:
    """A supplied verifier/strategy in event form. See docs/protocol-events.md."""
    events: tuple[Event, ...]

    @classmethod
    def from_tables(cls, protocol: TwoRoundProtocol, strategy: LocalStrategy) -> "TwoRoundProgram":
        """Spell out the standard schedule without traversing the supplied tables."""
        if not isinstance(protocol, TwoRoundProtocol) or not isinstance(strategy, LocalStrategy):
            raise TypeError("Expected TwoRoundProtocol and LocalStrategy")
        c, x, y, v, z = map(ClassicalRegister, ("case", "x", "y", "v", "z"))
        r = QuantumRegister("R_keep", _table_dimension(protocol.reference_dimension))
        q1 = QuantumRegister("Q1", _table_dimension(protocol.message_dimension))
        q2 = QuantumRegister("Q2", _table_dimension(protocol.second_message_dimension))
        m = QuantumRegister("M", q1.dimension)
        return cls((
            Prepare(c, x, r, q2, q1, protocol.cases),
            Deliver(x), Deliver(q1), Instrument(x, q1, y, m, strategy.first),
            Reply(y), ChooseContinuation(c, y, v), Deliver(v), Deliver(q2),
            Measure((x, y, v), (q2, m), z, strategy.final), Reply(z), Score(c, y, z, r),
        ))

    def lower(self) -> tuple[TwoRoundProtocol, LocalStrategy]:
        """Check bindings, transfers and local reads; erase them to the existing dialect.

        This is structural admission, *not* validation of the mathematical data.
        replay applies its normal table, exact-physics and resource checks.
        """
        if type(self.events) is not tuple:
            raise TypeError("Program events must be a tuple")
        if len(self.events) != len(_ORDER):
            raise ValueError(f"A two-round program requires exactly {len(_ORDER)} events")
        for i, (event, kind) in enumerate(zip(self.events, _ORDER, strict=True), 1):
            if type(event) is not kind:
                raise ValueError(f"Event {i} must be {kind.__name__} in a two-round program")
        p, d1, d2, ins, reply1, follow, d3, d4, measure, reply2, score = self.events

        # The input/output roles bind names; merely giving a slot a familiar name
        # never makes it visible. Forbid name reuse even across classical/quantum kinds.
        classical = (p.private_case, p.question, ins.answer, follow.question, measure.answer)
        quantum = (p.reference, p.second, p.first, ins.memory)
        if (any(type(b) is not ClassicalRegister for b in classical)
                or any(type(b) is not QuantumRegister for b in quantum)):
            raise TypeError("Bindings require classical and quantum registers of the declared kind")
        names = [r.name for r in (*classical, *quantum)]
        if len(set(names)) != len(names):
            raise ValueError("Each bound register name must be fresh")
        if ins.memory.dimension != p.first.dimension:
            raise ValueError("Instrument memory must have the first message dimension")

        _deliveries(d1, d2, p.question, p.first, "First")
        _reads((ins.question, ins.message), (p.question, p.first), "First local instrument")
        _reads((reply1.answer,), (ins.answer,), "First reply")
        _reads((follow.private_case, follow.first_answer), (p.private_case, ins.answer),
               "Verifier continuation")
        _deliveries(d3, d4, follow.question, p.second, "Second")
        _reads(measure.history, (p.question, ins.answer, follow.question), "Final local history")
        _reads(measure.wires, (p.second, ins.memory), "Final local POVM")
        _reads((reply2.answer,), (measure.answer,), "Final reply")
        _reads((score.private_case, score.first_answer, score.final_answer, score.reference),
               (p.private_case, ins.answer, measure.answer, p.reference), "Verifier score")

        if any(type(t) is not tuple for t in (p.cases, ins.moves, measure.moves)):
            raise TypeError("The case and move tables must be tuples")
        return (TwoRoundProtocol(p.reference.dimension, p.first.dimension, p.cases,
                                 second_message_dimension=p.second.dimension),
                LocalStrategy(ins.moves, measure.moves))
