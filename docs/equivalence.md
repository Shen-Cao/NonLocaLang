# Checked relabellings and evidence transport

`nonlocalgames.equivalence` implements invertible renaming of parties, local
questions, and local answers. Answer permutations may depend on the party's new
question. Every map has a source `Scenario` and a derived target `Scenario`.
This is a change of interface coordinates; a party renaming does not represent
a physical swap channel between separated laboratories.

Local transformations have a broader theory, including irreversible operations
and alphabet-changing liftings; see [Rosset et al.](https://arxiv.org/abs/2004.09405).
This primitive covers only bijective relabellings. It does not quotient by affine
score changes, nonsignalling-null expressions, discarded promise rows, or general
preprocessing and postprocessing.

## Direction and composition

All descriptor maps run **from new labels to old labels**. For new party `i`,
`party_permutation[i]` is the corresponding old party. `input_maps[i][x]` is its
old question, and `output_maps[i][x][a]` its old answer. Omitted maps are identities.
The constructor rejects noninteger labels and maps that are not bijections.

```python
from nonlocalgames.games import chsh
from nonlocalgames.equivalence import Relabelling, relabel_game

game = chsh()
r = Relabelling(game.scenario, party_permutation=(1, 0),
                input_maps=((1, 0), (0, 1)),
                output_maps=(((1, 0), (0, 1)), ((0, 1), (1, 0))))
renamed = relabel_game(game, r)
assert relabel_game(renamed, r.inverse()) == game
assert r.then(r.inverse()) == Relabelling(game.scenario)
```

`r.then(s)` applies `r` first and then `s`; `s.source` must equal `r.target`.
Inverting an answer map also reindexes it by the inverse question map.
`source_input` and `source_output` expose the complete profile mapping.
Game transport keeps the sampling distribution and payoff table separate,
including payoffs at zero-probability questions and signed rewards. Names are
preserved unless explicitly overridden.

## Resources and witnesses

`relabel_resource` permutes local dimension bounds and fixed-state tensor factors,
renames PPT cuts, and renames source endpoints while retaining source order and
endpoint order within every source. Measurement and randomness restrictions stay
explicit. Symbolic dimension bounds do not allocate the corresponding Hilbert
space. `relabel_task` carries both the data and the resource of an `AnalysisTask`
or `NetworkCompatibilityTask`.

Witness operations are explicit: `relabel_deterministic_strategy`,
`relabel_behavior`, `relabel_quantum_strategy`, `relabel_exact_strategy`,
`relabel_classical_network_strategy`, and `relabel_quantum_network_strategy`.
Independent source laws/states remain in their original source-factor order;
local measurement and response tables follow the renamed parties and labels.
An arbitrary global state is never repackaged as independent source states.

Exact witnesses preserve exact Born probabilities and scores. Numerical witnesses
preserve their stored entries and tolerance, then undergo ordinary validation.
Changed floating-point evaluation order can move a boundary case outside the same
tolerance. `NumericalRelabellingError` surfaces that failure; no state repair or
tolerance increase is performed. Numerical score comparisons still need numerical
tolerances. A backend specialized to a particular labeling of a chain can cease to
appear in task capabilities after moving its center; witness transport does not
silently rewrite backend assumptions.

## Fresh evidence

`relabel_evidence(game, artifact, r)` accepts raw evidence for the supported
standard unrestricted resource models, verifies the source, builds a new target
certificate, and verifies the target. Its return value has `artifact` and
`receipt` fields. The fresh artifact can enter ordinary claim/property assessment;
the receipt records provenance and cannot establish a claim by itself.

| Evidence | Transport and verification |
|---|---|
| `local_strategy` | Rename response tables and recompute the attained score |
| `local_enumeration` | Rename the maximizing tables and independently replay enumeration on both games |
| `quantum_strategy` | Permute exact state factors and POVMs, then verify a fresh Born-score certificate |
| `xor_primal` | Apply signed party/question permutation to both Gram axes and recheck attained score |
| `xor_dual` | Permute party/question diagonal entries and recheck exact PSD on the target |
| `nonsignalling_lp`, `numerical_hint` | Explicitly unsupported |

For XOR duals, binary answer flips conjugate the Gram cost by a diagonal sign
matrix. They leave the dual diagonal unchanged; party/question permutations
reorder it. Nonsignalling equality-dual rows use a fixed reference question, so
arbitrary input relabellings do not merely permute those rows. Spiral certificates
likewise have orientation-dependent copy/row conventions. Neither format is
transported by editing a fingerprint. Restricted-resource witness transport is
available, but proof admission retains its documented standard-resource scope.

## Budgets and identity

`max_map_entries` bounds `n + sum(X_i) + sum(X_i*A_i)` before consuming map
iterables or materializing identity maps. Inverse maps inherit the caller's
budget; composition defaults to the smaller source budget. Table transport has
an independent `max_entries` limit. Numerical state/witness routines bound actual
matrix allocations; exact witness transport inherits the witness's `ExactBudget`
unless explicitly overridden. Exhaustion does not refute mathematical equivalence.

`relabelling_to_dict/from_dict` use a versioned data schema. `relabelling_id`
binds the exact map and source interface, excluding allocation policy. A map ID
is not a verifier ID or a canonical orbit identity. Transport receipts retain
source/target verifier, resource, and evidence IDs and the complete descriptor.
Canonicalization is a separate operation with its own completeness requirement.
