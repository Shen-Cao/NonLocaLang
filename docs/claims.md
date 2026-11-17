# Claims, evidence, and verification admission

The claim layer asks precise questions about a resource value, rather than
interpreting a solver's status as a mathematical answer. A value is a supremum;
a value of one does not by itself assert an attained perfect strategy.

`ValueClaim(resource, relation, threshold)` uses `at_least` or `at_most` and an
exact threshold. `GapClaim(higher, lower, threshold=0)` asks whether the difference
of the two resource values is strictly greater than a nonnegative exact threshold.
Floats and Boolean thresholds are rejected; use integers, Fractions, or rational
strings. These definitions do not assert pseudo-telepathy or self-testing.

## Truth and execution

| Field | Values | Interpretation |
|---|---|---|
| `truth` | `proved`, `disproved`, `unknown` | Mathematical verdict for the precise claim |
| `execution` | `completed`, `unsupported`, `budget_exceeded`, `invalid_evidence`, `conflict`, `checker_error`, `dependency_unavailable` | Outcome of the verification/inference procedure |

Only completed execution may return a proved or disproved verdict. An empty or
insufficient evidence set can complete successfully with an unknown verdict.
Conflicting checked bounds quarantine inference as unknown/conflict: no preferred
solver or convenient bound is selected to conceal the contradiction.

The initial inference rules use only the unrestricted standard models:

`local ⊆ quantum_finite ⊆ quantum_closure ⊆ quantum_commuting ⊆ nonsignalling`.

A lower bound propagates toward larger sets, and an upper bound toward smaller
sets. This implication compares suprema. A separate rule below handles finite
quantum value claims with dimension bounds. Fixed states, source networks,
PPT/projective restrictions, and nondefault randomness policies require other
checked rules. Numerical hints may
retain such a restricted source context without contributing mathematical bounds.

For an interval `[L,U]`, `value >= t` is proved by `L >= t` and disproved by
`U < t`; `value <= t` is proved by `U <= t` and disproved by `L > t`.
For a strict gap, proof needs `L_high - U_low > threshold`; disproof needs
`U_high - L_low <= threshold`. Equality at the threshold is not a strict gap.

## Data artifacts are not automatically proofs

`EvidenceArtifact` snapshots a versioned kind, exact verifier ID, complete source
resource representation, and data payload as immutable canonical JSON strings.
The resource and artifact IDs are recomputed on decoding. Duplicate fields,
nonfinite JSON values, changed IDs, unknown fields, and oversized envelopes are
rejected. Decoded payloads are fresh objects; modifying them cannot mutate the
artifact. Hashes establish content identity, not mathematical truth or authorship.

`verify_artifact` rechecks the payload against the actual game. The current registry is:

| Evidence kind | Source resource | Payload | Admitted facts |
|---|---|---|---|
| `local_strategy` | Standard local | `strategy`: deterministic response tables | Exact local lower bound only |
| `local_enumeration` | Standard local | `strategy`, exact `value` string | Lower and upper after independent exhaustive replay |
| `nonsignalling_lp` | Standard nonsignalling | Rational-string `primal` and `dual` arrays | Exact NS lower/upper feasibility bounds |
| `xor_dual` | Standard commuting quantum | Existing fingerprinted XOR dual certificate | Exact commuting upper bound, also applicable downward |
| `quantum_strategy` | Standard finite tensor quantum | Full rational quantum lower certificate | Exact achieved quantum lower score |
| `numerical_hint` | Recorded source context | JSON observation object | No certified bounds, regardless of asserted labels |

The equality-dual XOR certificate bounds commuting strategies for its supported
parity score. Declaring its source as nonsignalling would be a different claim
and is rejected. Conversely, an NS primal lower bound is never propagated into
the quantum sets. Canonical CHSH symbolic results are not automatically admitted
as general game evidence; theorem-specific binding/transport needs its own rule.

External `AnalysisResult` objects and externally constructed checker-output
objects are not evidence inputs. The `Guarantee` enum validates vocabulary and
scalar types, but an `exact` label does not establish a fact. Numerical hints
remain hints even if their payload falsely contains `guarantee="exact"`.

## Dimension-bounded thresholds

`assess_value` accepts a finite tensor-product POVM contract whose only added
restriction is one local dimension upper bound per party. A verified quantum
witness gives its attained score as a lower bound when every physical dimension
fits the corresponding cap. A checked deterministic local strategy can also be
realized using dimension one at every party. Upper bounds on the unrestricted
finite quantum value descend to this smaller set. A larger witness that fails
the fit test supplies no scoped lower bound. Contradictory unrestricted
evidence still quarantines the query before applying the dimension rule.

The dimension assessment and its interval bind the full target resource ID
and ordered dimension tuple. The receipt schema is
`nonlocalgames.dimension_claim_assessment.v1`. Restricted gap and
perfect-strategy queries retain their separate admission rules.

To replay a saved threshold or gap in a fresh process, use:

```sh
python -m nonlocalgames claim verifier.json definition.json evidence.json
```

Use the finite game codec for the verifier, `claim_to_dict` for the definition,
and a JSON array of `evidence_to_dict` envelopes. The reader checks the
definition's binding and replays each raw artifact under the supplied budgets.

## Example

```python
from fractions import Fraction
from nonlocalgames import ResourceSpec
from nonlocalgames.analysis import classical_value, certify_nonsignalling
from nonlocalgames.claims import EvidenceKind, GapClaim, assess_gap, make_evidence
from nonlocalgames.games import chsh

game = chsh()
local = classical_value(game)
ns = certify_nonsignalling(game)
evidence = [
    make_evidence(game, ResourceSpec("local"), EvidenceKind.LOCAL_ENUMERATION,
                  {"strategy": local.witness, "value": str(local.value)}),
    make_evidence(game, ResourceSpec("nonsignalling"), EvidenceKind.NONSIGNALLING_LP,
                  {key: list(map(str, ns.witness[key])) for key in ("primal", "dual")}),
]
claim = GapClaim(ResourceSpec("quantum_finite"), ResourceSpec("local"), Fraction(0))
result = assess_gap(game, claim, evidence)
assert result.truth == "unknown"  # NS value 1 does not prove quantum advantage.
ns_claim = GapClaim(ResourceSpec("nonsignalling"), ResourceSpec("local"))
assert assess_gap(game, ns_claim, evidence).truth == "proved"
```

`ValueClaim` uses the same evidence with `assess_value`. Returned intervals retain
the IDs supporting their tightest bounds. `checks` records checker identities,
versions, source models, and admitted bounds; numerical observations are separate.

`evidence_to_dict/from_dict` preserve raw evidence envelopes. `claim_to_dict` and
`claim_from_dict` bind a definition to its verifier; `assessment_to_dict` writes a
receipt. An assessment receipt is not a standalone proof: archive its definition
and raw artifacts, then rerun inference with the trusted checker.

Run `python scripts/claim_demo.py > output/claim-demo.json` for CHSH, GYNI, and
constant-score examples. The demo includes deliberately untrusted numerical
labels and records unknown quantum gaps rather than promoting them to theorems.

## Budgets and trust boundary

`VerificationBudget` limits artifact count, full-envelope bytes, local strategies,
NS variables, exact PSD matrix size, and supplied scalar bits/text lengths. See
[exact scalar data boundaries](scalar-wire.md) for grammar and explicit overrides.
These are count/size limits per artifact,
not a global wall-clock guarantee; rational coefficient bit complexity matters.
The caller chooses budgets, and payloads cannot override them. Exhaustion is not
a negative mathematical result. Wider execution isolation belongs to the later
batch workflow.

The trusted boundary includes game/resource semantics, exact model lowering,
resource-inclusion rules, parsing, arithmetic, and checker implementation. This is
independent certificate checking, not a proof-assistant formalization. Differential
and adversarial tests remain necessary, including tests of model compilers shared
between numerical discovery and exact verification.

Exact quantum evidence is described in [exact quantum witnesses](exact-quantum.md).
The quantum checker has explicit dimension, realification, matrix-entry, event,
party, and scalar-bit budgets. It can prove a strict quantum gap from a feasible
point and an independent classical upper bound; it does not claim global
optimality or self-testing.


`verify_evidence_bundle` is the shared raw-artifact admission path for value and
property claims. It records attained scores separately from bounds and retains
witness dimensions where known. A returned bundle is a receipt, not an input proof
token. Claim-assessment receipts use schema v2 for these additional checked facts;
claim-definition and evidence-envelope schemas remain unchanged. See
[properties](properties.md) for the narrowly supported attainment predicates.
