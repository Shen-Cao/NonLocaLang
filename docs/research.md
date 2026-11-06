# Research map

The theoretical report is built from `report/main.tex`. It separates five
semantic components: arena and information flow, verifier, admissible
resources, analysis question, and optional target realizations/equivalences.
The first executable fragment is finite, classical-interface, one-round games.
The report proposes a typed causal extension without claiming that this is
already implemented.

## Results established in the report

- Finite verifier tables preserve linear scores exactly. Sampling and payoff
  remain separate because experimental statistics depend on both.
- Local bijective relabelings preserve values; positive affine payoff changes
  have an explicit value and score-deficit transformation.
- Hidden mixtures obey an upper bound by the weighted component values.
  Near saturation bounds each component deficit by the total deficit divided
  by its mixture weight. Publicly revealing the component changes the game.
- Product witnesses provide a lower bound for parallel conjunction under
  suitable resource closure. Equality requires a separate theorem.
- A no-input scalar-score game cannot exhibit a resource advantage if every
  deterministic output tuple is admissible. This includes ordinary unrestricted
  Bell resources and constant-output strategies in independent-source networks.
  Network nonlocality can still be detected through distribution compatibility
  and nonlinear witnesses.
- CHSH has an explicit exact sum-of-squares upper certificate and a matching
  two-qubit realization. A generic numerical SDP result has a different
  evidence status.
- Maximally entangled states of local dimensions 6 and 10 both attain the
  optimal CHSH score and extract a Bell pair, even though neither can be
  converted into the other by product local trace-preserving channels. This
  demonstrates why pairwise convertibility of full physical realizations is
  not the criterion for a unique self-tested target.

These elementary propositions serve as compiler contracts. They are not
presented as new discoveries in quantum foundations. The research question is
whether such contracts support a useful reusable construction and analysis
language.

## Literature and the design decisions it constrains

| Source | Consequence for this project |
| --- | --- |
| [Cleve, Høyer, Toner, Watrous: nonlocal strategies](https://arxiv.org/abs/quant-ph/0404076) | The classical one-round verifier is the finite core; two-party XOR games admit a specialized vector SDP. |
| [Brassard, Broadbent, Tapp: multipartite parity games](https://arxiv.org/abs/quant-ph/0306042) | The uniform even-input GHZ benchmark has local value 3/4 and quantum value 1. |
| [Brassard, Broadbent, Tapp: pseudo-telepathy games](https://arxiv.org/abs/quant-ph/0407221) | The row/column magic-square convention has local value 8/9 and a perfect quantum strategy; other question formats can have different values. |
| [Almeida et al.: guess your neighbour's input](https://arxiv.org/abs/1003.3844) | The three-party even-input GYNI benchmark has local and quantum value 1/4, with non-signalling value 1/3. |
| [Navascués, Pironio, Acín: convergent SDP hierarchy](https://arxiv.org/abs/0803.4290) | Finite moment relaxations are outer bounds; the complete hierarchy targets commuting operators. |
| [Slofstra: quantum correlations are not closed](https://arxiv.org/abs/1703.08618) | Use a supremum and record attainment separately from the value. |
| [Ji et al.: MIP* = RE](https://arxiv.org/abs/2001.04383) | No complete, uniformly terminating analyzer for arbitrary entangled game values; finite-dimensional closure and commuting resources differ. |
| [Fritz: correlation scenarios](https://arxiv.org/abs/1206.5115) | Source independence is a semantic constraint even in experiments without external inputs. |
| [Wolfe et al.: quantum inflation](https://arxiv.org/abs/1909.10519) | Independent-source models need causal compatibility machinery, not an ordinary Bell relaxation with a different label. |
| [Buscemi: semiquantum games](https://arxiv.org/abs/1106.6095) | Quantum question states cannot be replaced by freely readable preparation labels. |
| [Gutoski and Watrous: quantum strategies](https://arxiv.org/abs/quant-ph/0611234) | Sequential interfaces need causal trace constraints and explicit local memory. |
| [Mayers and Yao: self-testing](https://arxiv.org/abs/quant-ph/0307205) | State and measurement identification requires a local extraction/equivalence notion. |
| [McKague and Mosca: generalized self-testing](https://arxiv.org/abs/1006.0150) | Complex conjugation and flagged simulations constrain measurement uniqueness. |
| [Bamps and Pironio: SOS and self-testing](https://arxiv.org/abs/1504.06960) | Algebraic saturation relations are useful proof primitives, but extraction needs additional reasoning. |
| [Kaniewski: robust analytic self-testing](https://arxiv.org/abs/1604.08176) | Robustness should be a quantitative extraction guarantee with an explicit fidelity convention. |
| [Šupić and Bowles: review](https://arxiv.org/abs/1904.10042) | Useful orientation on definitions and limitations; primary papers support the concrete methods above. |
| [Boghiu, Wolfe, Pozas-Kerstjens: Inflation](https://arxiv.org/abs/2211.04483) | A specialized future adapter is preferable to reimplementing causal compatibility from scratch. |

Public identifiers are also stored in `report/bibliography.bib`. Downloaded
copies may be cached in the ignored `references/` directory; none is required
to build or use the tracked project.

## Next research milestones

1. **Structure-preserving constructors.** Add linear-system, graph, and
   synchronous-game families. Compare symbolic description size, table size,
   compilation time, and backend cost. A tiny predicate can expand into an
   enormous table; retain it when a specialized solver can consume it.
2. **Independent certificate checking.** Extend the rational non-signalling
   LP certificate and symbolic CHSH certificate to SDP duals and more general
   SOS decompositions, then verify them with exact or interval arithmetic.
   Keep the discovery process outside the trusted checker.
3. **Composition of rigidity.** Establish conditions under which saturation
   relations from component games admit compatible local extraction maps.
   Track how mixture weights and dimension costs affect robustness.
4. **Independent-source compatibility.** Introduce source hypergraphs and a
   distribution-valued task, with an Inflation adapter and explicit handling of
   global shared randomness. Test that unsupported Bell backends are rejected.
5. **Trusted and sequential interfaces.** Add semiquantum states and causal
   channels as typed objects. Demonstrate a faithful lowering on one restricted
   example before generalizing.
6. **Experimental and cryptographic metrics.** Introduce explicit noise,
   sampling, detection, memory, and adversary models. Winning-probability gaps
   alone do not establish statistical significance or secret randomness.

General self-testing, a universal optimizing compiler, and complete quantum
value computation are not current capabilities. The meaningful progress
criterion is reusable definitions and checked deductions on clearly stated
fragments.
