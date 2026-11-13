# Precise structural and attainment properties

`PropertyClaim` separates a structural question about a verifier from a statement
about an attained strategy. `assess_property` returns proved/disproved/unknown and
an independent execution status. It does not use numerical rank, optimizer
uniqueness, or an asserted result label as proof.

| Kind | Meaning |
|---|---|
| `acceptance` | Selected payoff entries lie in [0,1] |
| `boolean_predicate` | Selected payoff entries are zero or one |
| `xor_score` | Every party has binary outputs and each selected input's reward depends only on global answer parity |
| `xor_predicate` | Every party has binary outputs and exactly one global answer parity wins at each selected input |
| `perfect_strategy` | An admissible local or finite quantum strategy attains winning probability one on sampled play |
| `pseudotelepathy` | A perfect finite quantum strategy exists and no ordinary local strategy is perfect |

Structural kinds take no resource parameter and default to `off_support="all"`.
Set `off_support="sampled"` to ignore zero-probability input rows. Both definitions
retain the complete verifier identity, so changing an unused row changes the bound
context even when the sampled property answer is unchanged. XOR recognition covers
any finite party count; it does not imply applicability of the bipartite XOR SDP.
Constant and signed parity scores can be XOR scores without being XOR predicates.

Perfect-strategy claims require an explicit `ResourceSpec`. The initial fragment
supports only standard unrestricted local and finite tensor-product quantum models.
Pseudotelepathy has those two models fixed by definition. Both kinds require
acceptance rewards on the sampled support, and do not accept an off-support policy
that would reinterpret winning probability one as winning unused questions.

## Bound evidence and attained points are different

The shared evidence verifier records `attained_score` only after checking an actual
local table, NS primal behavior, or exact quantum realization. Its source model
remains explicit. A perfect NS behavior is not a perfect finite quantum strategy.
An upper certificate carries no attainment fact, and a numerical hint carries no
certified bounds or attainment even if its payload claims exactness.

Finite quantum perfectness is proved by a checked finite quantum score-one point
or a checked perfect local point with its elementary finite quantum embedding.
It is disproved by a checked quantum upper bound strictly below one. A supremum
interval [1,1] without an admissible attained point does not establish perfectness
in this inference fragment. Classical perfectness similarly uses an actual local
point or an excluding upper bound.

Pseudotelepathy needs finite quantum attainment and a local upper bound below one.
A checked perfect local point refutes it, as does a quantum upper bound below one.
Otherwise the verdict remains unknown. Rational approximation alone is not an
attainment proof. Self-testing and general family-recognition claims are outside
this schema and require their own definitions and verification rules.

Any contradictory checked bounds are quarantined. The acceptance premise also
excludes impossible lower bounds above one, upper bounds below zero, and attained
scores outside [0,1]. Loose upper bounds above one or lower bounds below zero are
not contradictions. An execution failure never becomes a mathematical disproof.

## API and command line

```python
from nonlocalgames import ResourceSpec
from nonlocalgames.games import ghz
from nonlocalgames.properties import PropertyClaim, assess_property

game = ghz()
syntax = assess_property(game, PropertyClaim("xor_predicate", off_support="sampled"))
assert syntax.truth == "proved"
perfect = PropertyClaim("perfect_strategy", ResourceSpec("quantum_finite"))
assert assess_property(game, perfect).truth == "unknown"  # No attaining evidence supplied.
```

Evidence-backed assessment accepts raw `EvidenceArtifact` objects and rechecks them
through `verify_evidence_bundle`. It never accepts an externally supplied bundle,
checker output, or property receipt as a proof token. Existing local-enumeration
evidence may trigger budgeted exhaustive replay; no unsolicited SDP or strategy
search is performed. Structural queries require an empty artifact iterable and
inspect only the bound verifier.

```sh
python -m nonlocalgames property ghz --kind xor_predicate --off-support sampled
python scripts/property_demo.py --evidence-dir output/property-evidence > output/properties.json
python -m nonlocalgames property ghz --kind pseudotelepathy --evidence output/property-evidence/ghz.json
```

The CLI accepts named benchmarks or `json --file ...`. Perfect-strategy queries
use `--resource` or a complete `--resource-file`; there is no implicit quantum
resource default. Evidence files are lists of raw artifact envelopes. Definitions
and receipts have distinct versioned codecs; only replay establishes the verdict.

`VerificationBudget.max_property_entries` bounds the full dense verifier table,
regardless of support policy. Other evidence budgets retain their M0/M1 meanings.
Identity construction still reads canonical verifier/resource data; these are not
wall-clock guarantees. Unsupported restrictions and exhausted budgets remain
unknown with their operational reason.
