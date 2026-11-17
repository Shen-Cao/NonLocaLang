# Three-question XOR games with weighted matching questions

Alice and Bob receive questions in $\{0,1,2\}$ and answer bits. The winning
answer parity is one when the questions match and zero otherwise. The referee
samples each diagonal pair with probability $r=p/3$ and each off-diagonal pair
with probability $q=(1-p)/6$, for rational $0\le p\le1$. Both endpoint games
retain all nine payoff rows, including rows of sampling probability zero.

## Exact values and certificates

Write the winning probability as $(1+\beta)/2$. The optimal biases for local
and unrestricted finite quantum strategies are

$$
\beta_C=\max\!\left(|1-2p|,\frac{1+2p}{3}\right),\qquad
\beta_Q=\max\!\left(|1-2p|,\frac{1+p}{2}\right).
$$

Let $H=qJ-(q+r)I$, where $J$ is the $3\times3$ all-ones matrix. Its entries are
the question probabilities signed by the winning parity. For a fixed choice of
Bob's three deterministic signs $b$, Alice's best response has bias
$\lVert Hb\rVert_1$. If all three signs agree, this is $|1-2p|$. With one
minority sign, it is $3r+2q=(1+2p)/3$.

In the bipartite XOR vector characterization, the quantum bias is the maximum
of $\sum_{xy}H_{xy}\langle u_x,v_y\rangle$ over real unit vectors.
Cauchy–Schwarz bounds it by $3\lVert H\rVert_2$. The eigenvalues are
$(1-2p)/3$ on the constant vector and $-(1+p)/6$ on its orthogonal complement.
For $p\le1/5$ the all-ones Gram matrix achieves the bound. For $p>1/5$ choose
Bob's three vectors at angles of $120^\circ$ to one another and Alice's vectors
as their negatives. This *trine* realizes correlation $-1$ on matching
questions and $1/2$ on the others. Both Gram matrices have rational entries.
[Cleve, Høyer, Toner and Watrous](https://arxiv.org/abs/quant-ph/0404076)

The quantum branch switches at $p=1/5$ and the local branch at $p=1/4$.
There is a strict advantage throughout $1/5<p<1$. The maximum gap in winning
probability is $1/16$, attained at $p=1/4$. The replayable evidence consists
of deterministic local enumeration, an exact unit-diagonal PSD Gram matrix,
and a dual diagonal with every entry equal to $\beta_Q/12$. The ordinary
checkers verify these witnesses in the original score convention.

```python
from fractions import Fraction as F
from nonlocalgames import ResourceSpec
from nonlocalgames.claims import GapClaim, assess_gap
from nonlocalgames.discovery.diagonal_xor3 import diagonal_xor3_evidence, diagonal_xor3_game

p = F(1, 4)
game = diagonal_xor3_game(p)
claim = GapClaim(ResourceSpec("quantum_finite"), ResourceSpec("local"))
assessment = assess_gap(game, claim, diagonal_xor3_evidence(p))
quantum, local = assessment.intervals
assert assessment.truth == "proved"
assert quantum.lower == quantum.upper == F(13, 16)
assert local.lower == local.upper == F(3, 4)
```

The primal checker assigns a rank-$d$ Gram matrix a sufficient local Hilbert
dimension $2^{\lfloor d/2\rfloor}$ on each side via a Clifford realization.
This construction gives dimensions $(1,1)$ for the all-ones witness and $(2,2)$
for the trine. Minimal dimension would require a separate argument. At $p=1$,
for example, only diagonal pairs are sampled and deterministic opposite answers
already achieve the trine's score one.

## An asymmetric perturbation

Begin with $p=1/4$ and move $1/48$ of probability from question $(0,2)$ to
$(0,1)$. The trine still achieves $13/16$ because its off-diagonal winning
probabilities are equal. For any strategy, the change in its score is at most
$1/48$, so the symmetric optimum also gives an upper bound of $13/16+1/48$.
Generic primal and dual discovery can improve both bounds for the new game.

```python
import json
from dataclasses import replace
from fractions import Fraction as F
from nonlocalgames.analysis import classical_value, certify_xor_upper_bound, verify_xor_certificate
from nonlocalgames.certificates.xor_primal import verify_xor_primal_certificate
from nonlocalgames.certificates.xor_primal_discovery import certify_xor_lower_bound
from nonlocalgames.discovery.diagonal_xor3 import diagonal_xor3_game

original = diagonal_xor3_game(F(1, 4))
mu = list(original.distribution)
mu[1] += F(1, 48)
mu[2] -= F(1, 48)
game = replace(original, distribution=mu, name="Asymmetric three-question XOR")
options = dict(solver="SCS", eps=1e-8, max_denominator=1_000_000)
pair = {
    "primal": certify_xor_lower_bound(game, **options).witness,
    "dual": certify_xor_upper_bound(game, **options).witness,
}
saved = json.loads(json.dumps(pair))
lower = verify_xor_primal_certificate(game, saved["primal"]).value
upper = verify_xor_certificate(game, saved["dual"]).value
assert classical_value(game).value == F(37, 48)
assert F(13, 16) < lower <= upper < F(13, 16) + F(1, 48)
```

In one run using these settings, rounding the verified rational endpoints
outward to six decimals gave $[0.814471,0.814472]$. The endpoints and ranks
can change with the numerical candidate. Rechecking the saved certificates
uses exact arithmetic and establishes the achieved lower score and the upper
bound for that sample.
