# kfp-explore: implementation plan

A web page for exploring how k-mer parameters behave: how k, sparsity and the noise floor change
precision, recall and area under the precision-recall curve (AUPRC), and how that depends on index
size, read length, target length and the homologues in the database. It uses kfp's methods:
information bits, the noise floor, anchors and combination k-mers.

## Verdict

**Feasible, with one limit: part of the noise floor must be measured, not derived.** A prototype
of the model is in [`checks/model_check.mjs`](../checks/model_check.mjs). It was compared with the
tables in kfp's `docs/tree-index-plan.md`.

| Question | Result | Status |
| --- | --- | --- |
| Is it cheap enough to be interactive? | 7,680 chains (k × sparsity × identity × read length) in ~165 ms in plain JS on one thread | Yes |
| Anchor recall against k, sparsity and identity | Fitted on m 8 (RMSE 0.010). Predicts held-out m 16 with RMSE 0.024 | Yes, once conservation is modelled |
| The same with independent mutations | RMSE 0.081, much too steep at low identity | Not usable |
| Chance anchors per read pair | Composition alone is 1.2–1.5× under at k 6–8 and 3.5–3.8× under at k 9 | Yes, with a measured correction |
| Index size | Representative residues / m is within ~20% of the kfp table | Yes |
| Noise floor tail (chance against score bar) | Independent hits fall ~1,000× per 10 bits. Decoys fall 10–12× | **No: fit it to decoys** |
| Precision among homologues | Modelled below, not yet tested | **Unknown, the largest risk** |

Limits of this evidence:

- The fit has 4 parameters and 16 points from one dataset.
- Its held-out test only extrapolates sparsity.
- The conservation parameters are poorly identified. (0.8, 32, 10) and (0.9, 32, 40) fit about
  equally well, and 0.9 is the edge of the grid. Treat them as a calibration, not as biology.
- Extending from one read pair to a protein assuming independent reads overstates recall (0.979
  against a measured 0.943 at m 16, identity 0.9). Reads from one protein share its conservation,
  so this needs a per-target term.

## Three levels of fidelity

Each level checks the one before, and the page shows them on the same axes.

1. **Analytic (milliseconds).** Runs on every slider move. Covers all the scans and the optimiser.
2. **Browser simulation (seconds).** Builds a real index from a bundled protein sample, simulates
   reads and reversed decoys, and scores them against known truth. This shows where level 1 is
   wrong. `data/mini-slice-100k` in kfp is a likely source; a 10 k protein subset is a few MB.
3. **Calibration tables (static JSON).** Measured values level 1 cannot derive: decoy tails, the
   hot k-mer correction and homologue spectra. They come from kfp outputs such as `bench.tsv`, or
   from a script the user runs on HPC.

## The model

**Inputs:** k, sparsity m, score bar, read length, target length, database size, per-base
divergence, depth, scenario.

**Matches along a read.** A two-state chain over site class (conserved or variable, block length
ℓ, conserved share φ, rate ratio ρ), crossed with the current match run. It gives P(no anchor) and,
with a hit counter added, the distribution of sampled hits. A shared divergence per target carries
the correlation between reads.

**Score.** kfp's evidence: for h hits, S = Σ bits − (h − 1) · log2(windows), with bits = −log2 q0
− log2 g per key. Use a distribution of per-key bits, not one value: at k 5, two hits score about
32 bits, so a 30-bit bar sits right on a step. Recall at a bar is P(S ≥ bar | identity).

**Noise floor.** Expected chance groups per read at each bar:

- The level at the first hit comes from postings × windows × composition, times a measured hot
  k-mer factor.
- The tail is an empirical curve per (k, sparsity), scaled linearly in proteins as
  `query.decoy_bar` does.
- Independent hits fail because of overlapping k-mers, biased local composition and sticky leaves
  (leaves that attract many chance hits). Key-count dispersion matters as well.

**Homologues and precision.** A spectrum R(x, N) gives the number of other groups at identity x to
a read's source, at database size N. Then:

- Precision(bar) = TP / (TP + Σ R(x, N) · P(S ≥ bar | x) + chance).
- Sweeping the bar gives the PR curve, and AUPRC follows as average precision.
- Presets, starting from the rarefaction table in kfp's plan: *close homologues*, *no close
  homologues*, *natural mix*.
- "Correct" must be selectable (leaf, group, unit), since it changes what counts as a false hit.
- Hits on a relative are treated as independent of hits on the truth. They are not, so level 2
  must test this first. kfp found that at scale the limit is assignment among close groups, not
  detection, so this is the term that matters most for large indexes.

**Index size.** postings = Σ max(residues / m, floor keys) over representatives, times bytes per
posting (~6 B for the anchor design, 19–20 B for today's tier). Representatives grow with the
database at a local exponent of ~0.6.

**Lengths.** Windows are limited by min(read, target) − k + 1. Reads per target are
Poisson(depth × L / read length), and short targets get kfp's floor, t = max(t_base, n_min / (L − k
+ 1)). Results are broken down by target length bin.

**SNPs in inconvenient places.**

- SNP to residue: about 3 × SNP rate × ~0.75 non-synonymous, then fed to the chain, with read
  errors added.
- For a given placement of keys on a read, the page shows the probability that d uniformly placed
  SNPs remove every key, and the fewest SNPs that can do so (a hitting set over key intervals).
- It also shows P(the read overlaps no key at all), which depends only on key spacing.

**Combination k-mers.** n k-mers within a span, hashed as one feature, then sampled at 1/m. Presence
is the chain's probability that all the windows match. Chance falls to about q^n × placements.
Exact any-feature statistics need more state, so use an approximation checked against level 2.
kfp's measured triples are the reference.

## Technology

- **Numeric core: Rust compiled to WebAssembly.** One implementation covers the model and the
  simulator, with native tests. Plain JS is already fast enough for level 1, so level 2 is what
  really justifies wasm.
- **Pyodide: not used.** It adds a large download and a slow start, and Python loops would be far
  slower than wasm. These are expectations, not measurements. It stays an option for a one-off
  analysis panel.
- **Pyrodigal: not used for now.** It is a gene caller, not a Python-in-the-browser runtime, so
  "pyrodigal" may have been meant as Pyodide. It only becomes relevant if the page ever starts from
  DNA with gene calling.
- **Do not depend on kfp's `crates/core`.** It pulls in needletail and uses `std::thread`, which
  does not work on plain `wasm32-unknown-unknown`. Copy `hash_kmer` and the alphabet, with test
  vectors checked against kfp's output.
- **UI: TypeScript, no framework, no bundler, Plotly** for hover, log axes and heatmaps. Simulation
  runs in a Web Worker. Parameters live in the URL so a view can be shared.
- **Hosting: GitHub Pages.** Static, with no backend.
- **Toolchain to install:** Homebrew's Rust has only the host target, so `rustup` with
  `wasm32-unknown-unknown` and `wasm-pack` are needed.

## Phases

Phases 1–2 are the plan. The rest is a backlog, reordered as questions come up.

**Phase 1: the first page.**

- Rust chain with `model_check` ported as native tests.
- Controls: k, m, read length, identity.
- Plots: recall against identity, and recall against m for several k.
- Readouts: index size at a chosen database size, and chance anchors per read.
- *Done when:* the check's tolerances hold natively, and slider moves redraw with no visible lag.

**Phase 2: browser simulation.**

- Reduced index from a bundled sample, simulated reads at a set identity, reversed decoys.
- Measured points overlaid on the model's curves.
- *Done when:* the simulation agrees with kfp for a matching configuration, using reduced-scale
  numbers from `results-hpc`. This phase also tests the independence assumption for homologues.
  If the model cannot follow the simulation on precision, level 2 becomes the only source for
  precision plots, at seconds per point rather than milliseconds.

**Phase 3: score and noise floor.** Hit-count distribution, per-key bits, decoy tail tables, a
score bar slider, the PR curve and AUPRC under one scenario.

**Phase 4: scenarios and scale.** Homologue spectra and presets, a database size axis, comparison
of small and massive indexes, and target length breakdown.

**Phase 5: SNP and overlap panel.** The miss probabilities above, with a key-placement diagram.

**Phase 6: optimiser.** Heatmaps of AUPRC against index size over k × m. The best parameters per
scenario, under an index size budget.

**Phase 7: combination k-mers.** Added to the model and the simulator.

**Phase 8: calibration script.** Turns kfp outputs into the JSON tables, and is shipped for the
user to run. Anything at full MGnify scale runs on HPC by the user.

## Open decisions

1. **Licence.** The repo has none, and it is public. Code copied from kfp (GPL-3.0-or-later) means
   this repo must be GPL-compatible, so GPL-3.0-or-later is the natural choice.
2. **Rust from phase 1, or TypeScript until phase 2?** The default here is Rust from phase 1, as
   requested, so there is only one numeric implementation.
