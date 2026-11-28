# Typed events for two-round replay

Use an event program to specify who can read a register at each move. For
example, the two halves of a Bell state have the same dimension, but transferring
the **second** wire before the first answer is an error. The checker distinguishes
the wires by their bindings, even when their matrices would fit either slot.

An existing verifier/strategy pair can be inspected as events without changing
its replay result:

```python
from fractions import Fraction
from nonlocalgames.protocols.interactive import TwoRoundProgram, replay
from nonlocalgames.protocols.interactive.examples import split_bell_label

protocol, strategy = split_bell_label("dephase")
program = TwoRoundProgram.from_tables(protocol, strategy)
report = replay(program)
assert report == replay(protocol, strategy)  # includes every leaf and both masses
assert len(report.transcripts) == 16
assert report.acceptance_probability == Fraction(1, 2)
```

## Writing the events directly

The following program uses the same numerical tables. The [table
API](interactive-replay.md) describes how to construct states, instruments, POVMs
and terminal effects. No event takes a Python callback.

```python
from nonlocalgames.protocols.interactive.events import (
    ChooseContinuation, ClassicalRegister, Deliver, Instrument, Measure,
    Prepare, QuantumRegister, Reply, Score,
)

secret, x, y, v, z = map(ClassicalRegister, ("secret", "x", "y", "v", "z"))
r, late, early, mem = (QuantumRegister("ref", 1), QuantumRegister("late", 2),
                       QuantumRegister("early", 2), QuantumRegister("mem", 2))

program = TwoRoundProgram((
    Prepare(secret, x, r, late, early, protocol.cases),  # verifier: r ⊗ late ⊗ early
    Deliver(early), Deliver(x),                         # verifier -> prover
    Instrument(x, early, y, mem, strategy.first),       # consumes early; retains mem
    Reply(y),                                           # prover -> verifier
    ChooseContinuation(secret, y, v),                   # cases[secret].after_first[y]
    Deliver(v), Deliver(late),
    Measure((x, y, v), (late, mem), z, strategy.final), # prover: late ⊗ mem
    Reply(z),
    Score(secret, y, z, r),                             # verifier: accept[z] on r
))
assert program.lower() == (protocol, strategy)
assert replay(program) == report
```

**Bindings and transfers.** Bind each of the nine slots once, with a unique
ASCII identifier of at most 64 characters. Direct quantum register declarations
use positive Python `int` dimensions. `from_tables` also accepts the legacy
positive integral dimension types (e.g., `numpy.int64`), converting them to
`int` without setting a replay budget. Booleans and floats are not dimensions.
The role of each slot comes from where it is bound. All initial wires belong to
the verifier. A classical delivery or reply reveals a value to its recipient and
preserves the sender's copy. Delivering a quantum wire moves ownership, without
copying it, reordering tensor factors or replacing it with its reduced state.
The instrument consumes the first wire and binds a same-dimension prover memory.
Only the reference remains with the verifier after the second delivery.

**Order and visibility.** The eleven events above are the whole dialect. In each
delivery pair, the classical and quantum deliveries may swap order because there
is no operation between them. No other events may be inserted, reordered or
omitted. Even a dimension-one second message has a delivery in the program.
The first instrument is chosen by the delivered first question alone. The
terminal POVM is chosen by the three *local* classical registers in order:
first question, own first answer, second question. The case, reference,
unreceived messages and other verifier data cannot be extra local inputs. A
verifier may still reveal a private bit through its *declared* question or
encode information in the prepared quantum state. The transcript returned to the
analyst contains the private case index; it is not a transcript handed to the prover.

**What is checked when.** `program.lower()` checks the fixed event grammar,
bindings, wire dimensions at use sites, and register access for both participants.
It returns `(TwoRoundProtocol, LocalStrategy)` using the three supplied table
tuples without scanning or copying their contents. Lowering alone makes no
claim about valid table rows, physics, or budgets. `replay(program, limits=...)`
lowers first and then runs the same table admission, size guards and exact
arithmetic as `replay(protocol, strategy, limits=...)`. Its budget for transcript
leaves is independent of the eleven syntax events. No second evaluator, probability
renormalization, or postselection is involved.

These checks constrain representable strategies in this trusted Python data
model. They do not isolate Python construction code or prevent arbitrary caller
side channels. This dialect has no general event interpreter, mid-protocol
verifier channel, multiple independent provers, exchange format, optimizer or
circuit-depth bound.
