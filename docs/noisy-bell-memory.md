# What a split-Bell readout misses

Two different memory channels can give the same **entire Bell-outcome table**,
not just the same success probability. The trusted [split-Bell verifier](interactive-replay.md)
prepares a uniform private label `g = 0,1,2,3` for Φ⁺, Φ⁻, Ψ⁺, Ψ⁻. It sends
Q1, waits for a fixed acknowledgement, then sends Q2. The channel acts on Q1
alone and stores its output in M. The Kraus index is not reported. Both
classical questions are constant; every label uses the same prover history
and final POVM. The acceptance rule, unlike the strategy, can use the label.

| First operation | Guess the Bell label | Read Z on M, accept + | Read X on M and Y on Q2, accept product χ(g) |
| --- | ---: | ---: | ---: |
| `damping` | 81/169 | 313/338 | 1/2 |
| `pauli_twirl` | 81/169 | 1/2 | 1/2 |
| `phase_plus` | 68/169 | 313/338 | 17/26 |
| `phase_minus` | 68/169 | 313/338 | 9/26 |

Each column is a **different** verifier/readout pair. Within a column, only
the first operation changes. These numbers are exact scores, not optima.

```python
from fractions import Fraction as F
from nonlocalgames.protocols.interactive import replay
from nonlocalgames.protocols.interactive.noisy_memory import noisy_split_bell_label

def run(channel, readout):
    return replay(*noisy_split_bell_label(channel, readout=readout))

bell = run("damping", "bell")
assert len(bell.transcripts) == 16
assert bell == run("pauli_twirl", "bell")        # both mass fields of every leaf
assert run("damping", "memory_z").acceptance_probability == F(313, 338)
assert run("pauli_twirl", "memory_z").acceptance_probability == F(1, 2)
for channel, score in [("phase_plus", F(17, 26)), ("phase_minus", F(9, 26))]:
    report = run(channel, "xy_parity")
    assert report.acceptance_probability == score
    assert sum(t.probability for t in report.transcripts if t.final_answer == 0) == F(1, 2)
```

## Why the readouts disagree

The damping channels have two Kraus operators, **diag(1,a)** and
**(12/13)|0⟩⟨1|**. The real channel uses a = 5/13; the conjugate channels use
a = (3 ± 4i)/13. All three have |a|² = 25/169. The fourth channel is the
Pauli twirl of the **real** channel: apply I, X, Z or Y with respective
weights (81,36,16,36)/169. The verifier does not sample or reveal a Pauli label.

* **Bell readout.** Given g, the probabilities for `z XOR g = 0,1,2,3` are
  |1+a|²/4, |1−a|²/4, 36/169, 36/169, respectively, for each damping
  channel. The jump changes the computational parity and contributes equally
  to the last two outcomes. Thus real damping and its twirl give the same row
  (81,16,36,36)/169 for the relative label; both phased channels give
  (68,29,36,36)/169. Multiply by the prior 1/4 for the joint transcript mass.
  Only z = g is accepted.
* **Stored-wire Z.** Q1 alone starts at I/2 for every label. All three
  damping channels send it to diag(313,25)/338, while the Pauli mixture
  preserves I/2. The other wire Q2 is still I/2 in each case: measuring Z
  there instead would miss the difference. Answer 0 denotes + and always
  passes this diagnostic; it is not a Bell-label guess.
* **Phase parity.** The fixed observable in replay's **Q2 ⊗ M** order is
  **Y ⊗ X**, with Y|0⟩ = i|1⟩. Set χ(g) = (+1,−1,−1,+1). For example,
  after a no-jump on Φ⁺ the unnormalized ket is (|00⟩ + a|11⟩)/√2:
  its contribution to the Y⊗X expectation is Im(a). The jump has zero
  cross term. Repeating for the other Bell kets gives expectation
  χ(g) Im(a). Therefore the conditional probability of outcome sign s is
  (1 + s χ(g) Im(a))/2. The raw positive sign is fair after pooling labels,
  but it matters which **private row** it came from: accepting the sign χ(g)
  scores (1 + Im(a))/2. For the real channel and its twirl both signs are fair
  even within a row.

## Scope

These readouts distinguish the specified channels with trusted preparations
and measurements; they are not tomography, a general memory witness, or a bound
on arbitrary POVMs. In particular the classical-memory ceiling of 1/2 for
guessing the **full** Bell label does not apply to the binary parity verifier.
A different strategy can measure Y on Q1, keep only the classical sign,
measure Y on Q2, and report minus the product. Since χ(g) is the eigenvalue
of −Y⊗Y for that Bell state, this strategy wins the parity test perfectly.
