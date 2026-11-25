# Exact scalar data boundaries

Portable game, claim, rational behavior, network strategy, and certificate readers
bound semantic rational data before converting strings or performing exact
arithmetic. Envelope byte limits alone are insufficient: a short string such as
`1e1000000000` describes an enormous integer. Scalar budget exhaustion rejects the
input; it never supplies a mathematical verdict or a numerical fallback.

`nonlocalgames.core.scalar_wire.parse_rational` accepts integers, Fractions, and
ASCII rational strings. It rejects Boolean, floating-point, and complex values.
Its default limits are 4096 bits per supplied integer component and 8192 text
characters. Callers can explicitly raise `max_bits` and `max_chars`. Public readers
and nonquantum certificate checkers expose these as `max_scalar_bits` and
`max_scalar_chars`; count, dimension, and file-byte limits remain separate.

The legacy text grammar accepts a leading sign, outer ASCII whitespace, underscores
between digits, whitespace around a slash, finite decimals, and scientific
notation. Examples include `-3/4`, `+0.75`, `1 / 4`, and `7_5e-2`. Fraction
denominators are unsigned and nonzero. Unicode digits, nonfinite values, and
arithmetic expressions are invalid. A zero denominator raises `ValueError`.

Checks precede integer conversion, decimal powers, and Fraction construction.
Component and power limits are conservative: a reducible noncanonical spelling
may exceed the budget even when its reduced value fits. Decimal trailing zeros
can be cancelled textually. Excessive exponents are rejected even when the
mantissa is zero. `ScalarBudgetExceeded` is a `ValueError` subclass that identifies
text or component budget exhaustion.

The exact quantum codec uses the same implementation with stricter canonical
syntax: a reduced integer or fraction, without extra signs, whitespace, leading
zeros, decimals, or exponents. It retains `ExactBudget.max_scalar_bits` and maps
scalar exhaustion to `ExactBudgetExceeded`; its text allowance scales with that
explicit bit budget.

Writers emit reduced `n` or `n/d` text. `scalar_string` uses bounded decimal chunks
so an explicitly raised bit budget does not depend on Python's process-wide
integer-to-string digit setting. Ordinary canonical output bytes, verifier
identities, and existing certificate fingerprints are unchanged. The formatter
itself has no size limit: callers must bound untrusted data first.
If an exact computation produces a larger stored bound or coefficient, decoding
that certificate may require a larger explicit scalar budget than its inputs did.

## Admission and scope

`VerificationBudget.max_wire_scalar_bits` and `max_wire_scalar_chars` apply to
nonquantum evidence admission. Quantum evidence retains its separate quantum
budget. Exceeding either yields `unknown` with `budget_exceeded`; malformed
semantic scalars yield `unknown` with `invalid_evidence`. Numerical hints and
descriptive metadata are not recursively interpreted as rational numbers.

Game JSON readers also check scenario/table sizes before scalar conversion.
`from_dict` accepts `max_entries`; task decoding calls this `max_game_entries`.
`load` additionally bounds file bytes. The property CLI exposes
`--max-scalar-bits` and `--max-scalar-chars` for game loading and nonquantum evidence.

### Reading JSON files

Use the file codecs for untrusted documents. They reject repeated names in
any object (even equal values or different JSON escapes for the same name),
and the nonstandard tokens `NaN`, `Infinity`, and `-Infinity`. This applies to
the verifier, task and network-compatibility loaders and to every input file
of the `claim` and `property` commands. A malformed file is an input error,
not a mathematical verdict. A `from_dict` codec cannot detect names already
discarded by another parser.

File limits count original bytes before decoding. The three loaders and
`property` default to 64,000,000 bytes per file and retain Python's JSON byte
encoding detection (UTF-8 with or without BOM, UTF-16, or UTF-32). Use UTF-8
without BOM for portability: `claim` and the discovery readers require it.
`claim` defaults to 8,000,000 bytes per verifier, definition, or evidence file;
its separate budget file is limited to 16,384 bytes. Neither the canonical
representation nor the content IDs of valid records depend on file encoding.

### Computational limits

Direct trusted Python constructors retain unrestricted exact arithmetic. These
wire limits bound supplied data, not every later product, Schur complement, hash,
or execution time. Exact quantum arithmetic has additional intermediate scalar
checks; other exact solvers can still grow larger intermediate rationals. Public
raw PSD verification bounds supplied entries, while prepared internal exact
matrices bypass text parsing without accepting strings or floats. Process-level
time and memory isolation belong to the execution layer.
