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
| Protocol semantics checkpoint | A family of prepare–route–measure experiments sharing one local strategy | Locally identical views impose identical responses across hidden wirings, including when a sent port returns to its original owner |
| M3: discovery loop | Bounded family specifications, enumeration, deduplication, screening, search, and verified claims | Restarting after interruption recovers the same coverage, checked verdicts, unresolved cases, and raw evidence. Run records bind the revision, family, resources, seeds, options and budgets |
| M4: reusable execution and ledger | Compiled-template caching, batch execution, budgets, and searchable evidence records | Measured reuse, isolated solver state, and recheckable provenance for every claimed result |
| M5: targeted scientific coverage | A scientific extension selected from measured needs of a structured family | Independent cross-checks and a verified translation back to the original mathematical task |
| M6: optional agent transport | A thin MCP or other transport adapter over the stable public operations | No duplicated mathematical logic, hidden solver fallback, or weakening of resource restrictions |

The M0 claim/evidence fragment is implemented for the standard unrestricted
resource chain; see [claims](claims.md). M1 now supplies
[exact quantum lower witnesses](exact-quantum.md). The
[narrow property layer](properties.md), [bounded scalar wire decoding](scalar-wire.md),
and [checked relabellings](equivalence.md) support the
[uniform XOR corpus](xor3.md). Its 512 member-specific gap claims replay from
raw representative evidence. The [protocol checkpoint](wiring.md) supplies a
bounded hidden-wiring arena, constructive classical witnesses, and an exact
cross-route marginal test. The first [M3 discovery run](weighted-chsh.md)
links numerical screening, rational witnesses, exact verdicts, and persistent
progress on the weighted CHSH grid. Exact replay is profiled using the
[reproducible workload](development.md#comparing-replay-cost). A
[weighted three-question family](diagonal-xor3.md) exercises rank-based
finite quantum attainment and matching rational upper certificates.

The [reusable XOR screen](weighted-chsh.md#reusing-the-numerical-screen)
binds the template and solver-state policy to each run. The remaining M4
ledger gate is cross-run search. A query spanning multiple recorded grids
should group matching verifier and claim IDs, identify overlapping coverage,
and distinguish replayed proof bounds, numerical candidates, and missing or
unresolved members. Stored evidence and checker provenance must be recoverable
for each reported proof.

The [dimension-bounded threshold rule](claims.md#dimension-bounded-thresholds) uses an attained finite-dimensional score
when the verified witness fits every dimension bound and satisfies all other target
restrictions. Value reductions for product or particular PPT contracts need
separate proofs. The [two-round replay](interactive-replay.md) checks a bounded
trusted-interface strategy without a new dependency. Broader NPA, quantum
inflation, and additional trusted/sequential interfaces are selected when a
discovery experiment requires them.

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

## Protocol semantics checkpoint

One witness supplies independent preparations before a hidden route is chosen and
operations indexed by the local information available after routing. Port identities
persist across routes, including a port that returns to its sender. The family ID
binds the routes, local views, timing, and preparation assumptions. A transcript
from one fixed route can be evaluated independently. Joint compatibility requires
the same admissible strategy across the entire family.

The first test uses singleton questions. In both cycles ABC and AC, Alice keeps
her own port and receives Charlie’s. With independent preparations, her marginal
must agree across these routes. A target requiring opposite deterministic outputs
at Alice in the two routes is individually feasible in each branch and jointly
impossible. A shared deterministic response gives a positive control. This
rational experiment isolates the information constraint in the
[fermionic rewiring experiment](https://arxiv.org/abs/2606.12363v1).
A [fixed six-mode reference](fermion-reference.md) now evaluates the graded
route action and algebraic measurements of the specified strategy. The full
experiment asks for joint behavior compatibility and requires a separate
exclusion argument.

## Discovery acceptance experiments

Start with exhaustive uniform three-input-per-party bipartite XOR predicates:
there are 512 raw predicates and three orbits under local input permutations,
input-dependent output flips, and party exchange. This is a deterministic
correctness corpus for equivalence and transport, not a claim of new games.

Replay each member against its exact verifier ID. Orbit sizes are 32, 288, and
192 in lexicographic representative order. The all-zero representative admits
perfect local play. Both other representatives have certified quantum advantages.
Archive raw evidence with sufficient information for fresh-process replay. For a
weighted control, biased CHSH with input-11 weight 1/10 has exact local and
quantum value 9/10, making it a regression for screening near zero gap.

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
