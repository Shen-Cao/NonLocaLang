# Uniform three-question XOR corpus

Each player receives one of three questions, with all nine joint questions
equally likely. A nine-bit mask specifies the winning answer parity for each
question pair in row-major order. The most significant bit describes `(0,0)`.
The family contains every such Boolean predicate, giving 512 exact verifiers.

From an installed checkout:

```sh
python -m nonlocalgames xor3 canonicalize 100000000
python -m nonlocalgames xor3 build output/xor3/corpus.json
python -m nonlocalgames xor3 replay output/xor3/corpus.json
```

`canonicalize` takes nine binary digits. The Python API uses integer masks:

```python
from nonlocalgames.discovery import canonicalize_xor3, xor3_game, xor3_mask
from nonlocalgames.equivalence import relabel_game

game = xor3_game(256)
result = canonicalize_xor3(game)
assert result.complete and result.representative == 1
assert xor3_mask(relabel_game(game, result.transporter)) == 1
```

The allowed transformations are independent local question permutations,
input-dependent answer flips, and exchange of the players. The canonical mask
is the least row-major bit string under this action. For each ordering of
questions and players, column flips can set the first row to zero. The remaining
row flips can set the first entry of every later row to zero. These choices
minimize the bits in lexicographic order and determine the whole resulting
table. Enumerating both player orders and both players' question permutations
therefore takes 72 candidates. Ties select the first minimizing order.
`--max-candidates` bounds this search. A limit below 72 returns
`complete: false` with null representative, orbit ID, and transporter.

## Claims and certificates

| Representative | Orbit size | Local optimum | Finite quantum evidence in the corpus |
|---|---:|---:|---|
| `000000000` | 32 | 1 | Perfect local play and a matching XOR upper bound |
| `000000001` | 288 | 8/9 | A two-qubit strategy attains 401/450 |
| `000001010` | 192 | 7/9 | A two-qubit strategy attains 187/225 |

Local upper bounds use exact enumeration. Each transferred point is scored on
the member's own verifier. For representative 10, the separate Python operation
`representative_evidence(10, tight=True)` supplies a strategy of local dimension
four and a dual with every entry equal to 1/18. The two certificates give the
same score, 5/6. Representative 1's displayed score is a feasible lower bound.

The archive includes the family and action versions, query, raw representative
artifacts, and one record per member. Each record binds its exact verifier and
claim, member-to-representative map, target evidence IDs, and checked scores.
Build writes the completed archive by atomic replacement. Replay reads the
archived representative witnesses, rebuilds each map, transports the evidence
in the inverse direction, and verifies the gap on the actual member. A valid
archive yields 480 proved and 32 disproved strict advantages. Incomplete
coverage or a member record differing from recomputation fails replay.

These identities concern uniform XOR predicates in this specific scenario.
Weighted samplers, other payoffs, restricted resources and hidden routing
protocols require their own family and query definitions.
