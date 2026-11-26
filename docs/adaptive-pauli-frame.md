# Adaptive Pauli-frame replay

This four-qubit example tests whether a [two-round replay](interactive-replay.md)
keeps the first Bell measurement, its reported bits, the later basis question
and the endpoint readouts consistent. All density matrices, projectors and probabilities
are rational; the example requires no optimizer or change to the replay limits.

```python
from fractions import Fraction
from nonlocalgames.protocols.interactive import replay
from nonlocalgames.protocols.interactive.pauli_frame import adaptive_pauli_frame

report = replay(*adaptive_pauli_frame())
assert report.acceptance_probability == 1
assert len(report.transcripts) == 32  # includes impossible final answers
assert [t.probability for t in report.transcripts if t.probability] == [Fraction(1, 16)] * 16
assert replay(*adaptive_pauli_frame(stale_right=True)).acceptance_probability == Fraction(3, 4)
assert replay(*adaptive_pauli_frame(scoring="zero")).acceptance_probability == Fraction(1, 2)
assert replay(*adaptive_pauli_frame(scoring="swapped")).acceptance_probability == Fraction(1, 2)
```

## Wires and messages

1. The verifier draws a uniform private bit **c**. It sends one prover both
   Φ⁺ pairs in register order **A, L, R, B** (A is the most significant bit).
   The pairs are AL and RB. The verifier keeps no quantum reference; the first
   classical question is always 0, and the preparation is independent of c.
2. The prover makes a Bell measurement on the inner qubits LR using four
   projective Kraus operators. It returns **y = 2p + q**, where p is the ZZ
   parity and q the XX parity (bit 0 is eigenvalue +1). Equivalently, a CNOT
   from L to R followed by H on L and Z readout gives the same outcome labels
   in order **q,p**. The fixture uses Bell projectors as Kraus operators,
   leaving LR in its measured Bell state. It does not compile that circuit.
3. After receiving y, the verifier sends the basis **b = c XOR p** (0 = Z,
   1 = X). The strategy row is selected by the delivered history (0,y,b).
   Each basis is marginally fair for each y, but the verifier-side transcript
   must satisfy b = c XOR p on every leaf.
4. The prover measures A and B in that basis and returns **z = 2a + d**
   (a from A, d from B). The verifier accepts if a XOR d equals p for Z or q
   for X. No qubit is physically corrected: the frame gives the expected parity
   in the chosen basis. The separate rule b = c XOR p chooses the challenge;
   the Pauli byproduct does not turn a Z measurement into an X measurement.

## An independent prediction for each leaf

The reduced state on LR before measurement is I/4, so each Bell label has
probability 1/4. For a given label the endpoint state is
`(|0,p⟩ + (−1)^q |1,1 XOR p⟩)/√2`, up to a global phase. It has ZZ eigenvalue
(−1)^p and XX eigenvalue (−1)^q; either endpoint by itself is maximally mixed.
For fixed c,p,q there are therefore two possible answer pairs, each with
unconditional mass (1/2)(1/4)(1/2) = **1/16**, and two zero-mass pairs.

| Change from nominal | What changes | Prediction |
| --- | --- | --- |
| `stale_right=True` | Only B measures in c instead of b. It reconstructs c from b XOR p after y; no private dispatch is allowed. | For p=0 both bases match. For p=1 the endpoints use crossed X/Z bases, giving four equiprobable pairs rather than two. Success is **3/4**. |
| `scoring="zero"` | Only the verifier changes: it now expects equal bits in either basis. | Fails exactly when the physical parity is 1. Success is **1/2**. |
| `scoring="swapped"` | Only the verifier changes: it uses q for Z and p for X. | Fails exactly when p differs from q. Success is **1/2**. |

The two scoring controls have identical sampling probabilities but distinct
failed transcripts. For p=q=1 the zero rule rejects both physical answers in
either basis, while the swapped rule accepts. A separate test flips only the
phase of the RB source (Φ⁺ to Φ⁻): all Z-basis readouts keep their parity,
but all X-basis readouts reverse it.

## What a perfect score does not show

The two endpoints belong to **one** prover. A deterministic prover that reports
y=0 and z=0 without measuring anything also wins; the tests replay that
counterexample. This is a supplied-strategy regression, not a self-test, proof of
quantum advantage or depth bound. The motivation is the use of Bell outcomes
to track Pauli corrections in §5.1 of [Hsieh et al., arXiv:2606.16425v1](https://arxiv.org/html/2606.16425v1).
The finite fixture is not that paper's delegated preparation protocol or its
two-round depth-hierarchy task. It imposes no spatial separation or circuit
light-cone constraints.
