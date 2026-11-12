# Discovery workflow milestones

The project is a resource-aware research workbench for constructing, comparing,
deduplicating, and independently verifying claims about nonlocal games. The next
capability is selected by its contribution to a defined discovery experiment or
to preventing an incorrect mathematical claim.

| Milestone | Deliverable | Acceptance gate |
|---|---|---|
| M0: claim and evidence semantics | Typed claims, bound directions, exact identities, verification admission, and distinct operational outcomes | External result labels cannot prove a claim; evidence is rechecked in its full context; failures are not mathematical falsehoods |
| M1: positive quantum certificates | Exact finite-dimensional state/POVM witnesses and independently checked lower scores | Strict quantum/classical separations can be proved; numerical repairs do not silently preserve fixed-state, projective, PPT, or network assumptions |
| M2: properties and transport | A narrow property layer, checked inference rules, and exact local-relabel transport | Structural properties and threshold claims have precise definitions; transported evidence preserves verifier and resource assumptions |
| M3: discovery loop | Bounded family specifications, enumeration, deduplication, screening, search, and verified claims | A run can be resumed with the same parameters, seeds, assumptions, and unresolved cases |
| M4: reusable execution and ledger | Compiled-template caching, batch execution, budgets, and searchable evidence records | Measured reuse, isolated solver state, and recheckable provenance for every claimed result |
| M5: targeted scientific coverage | One structured family and one backend adapter justified by an experiment | Independent cross-checks and a verified translation back to the original mathematical task |
| M6: optional agent transport | A thin MCP or other transport adapter over the stable public operations | No duplicated mathematical logic, hidden solver fallback, or weakening of resource restrictions |

The M0 claim/evidence fragment is implemented for the standard unrestricted
resource chain; see [claims](claims.md). M1 now supplies
[exact quantum lower witnesses](exact-quantum.md). M2 is the next queued milestone.
Dimension/PPT classical-value reductions
remain useful theorem-backed inference rules for M2. Broader NPA, quantum inflation,
and trusted/sequential interfaces are selected when a discovery experiment requires
them; their existence is not itself a reason to expand the dependency set.

## Mathematical verdicts

A claim records its verifier, resource model, quantifiers, threshold or target,
and the evidence used. Its verdict is proved, disproved, or unknown. Execution
status is separate: unsupported assumptions, exhausted budgets, invalid evidence,
and numerical failure must never become a false mathematical verdict.

For certified intervals L_Q <= omega_Q <= U_Q and L_C <= omega_C <= U_C,
the claim omega_Q - omega_C > epsilon is proved by L_Q - U_C > epsilon,
and disproved by U_Q - L_C <= epsilon. Otherwise it remains unknown.
An NS or NPA upper bound above the classical value is candidate screening only.

Perfect-strategy attainment is different from a supremum equal to one.
Pseudo-telepathy additionally requires that the classical value is below one.
Numerical optimizer uniqueness is not a self-testing proof. Self-testing claims
need a specified target, extraction equivalence, resource model, and proof.

The initial claim engine covers the unrestricted standard resource inclusion
chain. More restrictive resource contracts remain available to the analysis
backends, but claim transport must reject them until a checked rule applies.
Exact certificate checking relies on the documented mathematical compiler and
Python checker; it is not proof-assistant formalization.

## Identity and equivalence

Keep distinct identities for a verifier, resource contract, query, run, and
certificate. Exact verifier identity retains sampling and payoff separately,
including zero-probability entries; its descriptive name is not mathematical
content. An orbit representative requires a declared equivalence action and an
exact transporter. Party permutations must also transport dimension bounds,
fixed-state factors, PPT cuts, source graphs, and certification targets.

Canonicalization under a budget must report whether it completed. A partial
search or a convenient invariant is not an exact canonical representative.
Template caches include compiler/schema conventions. Result caches additionally
include resource, method/level, numerical options, seed, and code provenance.

## Discovery acceptance experiments

Start with exhaustive uniform three-input-per-party bipartite XOR predicates:
there are 512 raw predicates and three orbits under local input permutations,
input-dependent output flips, and party exchange. This is a deterministic
correctness corpus for equivalence and transport, not a claim of new games.

A larger discovery corpus must specify its family, rational parameter domain,
score normalization, sampling, and equivalence convention. General Boolean
winning tables in that scenario have 36 bits and are mostly not XOR games.
Weighted XOR families require an explicit weight grid. A leaderboard separates
proved separation, numerical candidates, and unresolved cases; a requested count
of successful proofs is not assumed to exist in the chosen corpus.

## Optional language backends

Python remains the public construction and verification interface. Structured
constructors can preserve algebra without introducing a new parser. Julia or
other backends are optional, isolated projects selected for concrete capabilities.
They communicate through versioned data with explicit solver selection, resources,
budgets, and provenance. A backend's numerical answer does not become a proof
unless an applicable independent checker accepts its evidence.
