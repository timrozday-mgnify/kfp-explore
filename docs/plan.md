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
| Analysing a small set in the browser | 44 k proteins: k-mers and spectrum in under 1 s | Yes, up to about 20 M residues |
| Extrapolating a small set to MGnify | 7× to 2,700× too low | **No: needs a measured scale profile** |

Limits of this evidence:

- The fit has 4 parameters and 16 points from one dataset.
- Its held-out test only extrapolates sparsity.
- The conservation parameters are poorly identified. (0.8, 32, 10) and (0.9, 32, 40) fit about
  equally well, and 0.9 is the edge of the grid. Treat them as a calibration, not as biology.
- Extending from one read pair to a protein assuming independent reads overstates recall (0.979
  against a measured 0.943 at m 16, identity 0.9). Reads from one protein share its conservation,
  so this needs a per-target term.

## The page and the measurements

They meet only through small versioned JSON tables. Each table records its source, sample and
parameters.

1. **The page, in the browser.** The analytic model runs on every slider move and covers all the
   scans and the optimiser. It reads the tables and computes nothing heavy.
2. **Measurements, run separately.** Python scripts and Rust binaries in `measure/`, run on a laptop
   for small samples or by the user on HPC at scale (Nextflow, SLURM). They produce:
   - inputs the model cannot derive: decoy tails, the hot k-mer correction, homologue spectra;
   - simulations with known truth: a real index, simulated reads and decoys, scored over a grid of
     parameters. The page overlays these points on the model's curves, which shows where the model
     is wrong.

A third piece joins them: the page can also analyse a small sequence set itself, and a scale
profile carries predictions up to full size. Both are described in
[Analysing a small set in the browser](#analysing-a-small-set-in-the-browser) and
[Scaling to full size](#scaling-to-full-size).

## Analysing a small set in the browser

Load the bundled mini-slice sample or your own FASTA, then compute directly with NumPy in Pyodide:
k-mer spectra, composition, lengths, rarefaction, reversed decoys, a sampled index, and simulated
reads scored against truth.

Measured in Pyodide under node, one thread, at k 7:

| Set | Residues | k-mers + unique + counts | Decoy lookup | Compressed |
| --- | --- | --- | --- | --- |
| 10 k representatives | 2.9 M | 0.3 s | 0.9 s | 1.7 MB |
| 44 k representatives | 12.6 M | 0.8 s | 14 s | 7.4 MB |
| 295 k proteins | 74 M | 5 s | 40 s | 13.9 MB |

- Lookup used per-query binary search. Sorting queries first or merging should be much faster, but
  that is untested.
- Peak memory over all three runs was 3.7 GB, close to wasm's 4 GB limit. The default should be
  10–50 k proteins, with a cap around 20 M residues.
- A sample this small cannot measure the rare tail at k 8 and above (see the pilot results).

## Scaling to full size

**A small set cannot be extrapolated to MGnify alone.** Tested by fitting units against proteins on
the mini-slice subsampled to smaller sizes:

| Fit | Predicts the slice (actual 55,433) | Predicts the release (actual ~1.8 × 10⁹) |
| --- | --- | --- |
| Power law, to 1/4 | 67,036 | 9 × 10⁷ |
| Power law, to 1/16 | 88,197 | 3 × 10⁸ |
| Curved in log-log, to 1/4 | 54,950 | 7 × 10⁵ |
| Curved in log-log, to 1/16 | 58,435 | 2 × 10⁶ |

Within the slice, the curved fit is within 1–5%. To the release, every fit is 7× to 2,700× low.
The slice is selected and dense, so its curve has a different shape from the whole database. The
share of proteins with no ≥ 90% relative shows the same pattern: extrapolating inside the slice
gives 0.147 against an observed 0.14, while the release-wide singleton share is 0.21.

**So measure a scale profile of MGnify, natively, and interpolate between its points.**

1. **A ladder of uniform samples:** 10⁵, 10⁶, 10⁷ and 10⁸ proteins, with the same statistics as the
   browser analysis. Sample proteins by a hash of `protein_id`. Sampling by cluster, as phase 0's
   `cluster_rep % sample` does, keeps whole clusters and so cannot measure how units grow.
2. **Whole-release anchors** where a full pass is cheap: proteins, MGnify90 units, singletons, and
   k-mer counts in hash slices (`stream.json`).
3. **Interpolate, do not extrapolate.** The page fits each statistic through the ladder and its
   anchor, as a curve against N.

| Statistic | Scaling rule |
| --- | --- |
| Representatives, postings | Profile curve × residues / m × bytes per posting |
| Chance anchors | Linear in postings (kfp's check), × hot k-mer factor from the profile |
| Noise floor tail | `--chance-scale` rule, linear in proteins with a geometric tail; sticky leaves from the largest rung |
| Homologue spectrum R(x, N) | Profile, per identity bin |
| k-mer occupancy at small k | 1 − exp(−postings × q): saturates, so never linear |

The page plots your set's curve and MGnify's together. Their ratio at the same N shows how your set
differs from a uniform sample, and a band shows the uncertainty.

*Check:* hide the largest rung, predict it from the others, and report the error. That error is
the band. Beyond 10⁸ only the anchors constrain the curve, so these predictions stay less certain.

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
- The score bar is the information bit floor, with its own control on the page.
- A starting estimate from kfp's two release figures (k 5 and k 6, by hand): chance groups per read
  at 30 bits are about 3–4 × 10⁻⁴ of chance anchors, falling ~10× per 10 bits. Other k are untested.
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

**Grouping and the group k-mer policy.** The index stores k-mers for groups, not for each protein,
so two choices change what a read can match:

- **Radius:** the identity within which proteins form a group. A wider radius gives fewer groups,
  and relatives inside it stop being false hits, but keys must represent more diverse members.
  kfp's groups are MGnify90 units or `tree:<d>` (0.34 ~ 95%, 0.57 ~ 90%, 0.83 ~ 80% identity), and
  `build.wide_clusters` makes wider ones.
- **Policy:** which member k-mers become the group's keys:
  - one representative (`medoid`, `central`, `longest`);
  - the union (`composite:0`);
  - a soft intersection, k-mers held by at least a share f of members (`composite:f`, today f = 0.25);
  - the top n by member frequency.

  Sparsity is applied after the policy, or as a key budget per group (`--layer-keys`).

In the chain, a policy sets two things per site class: the identity between the read and the key's
source, and the chance a matching window is kept as a key. Soft intersection and frequency ranking
move keys into conserved windows, which is why they keep more matches at low identity. The policy
also changes postings, because the union grows with group diversity and an intersection shrinks.

*First test (exploratory):* a class-only prediction of kfp's retention table (share of a group's 32
keys a held-out protein holds, at identity 0.6–0.9, four policies) gave RMSE 0.11, no better than
independent mutations (0.11). It overestimates soft intersection and frequency ranking, for example
0.82 against 0.52 for f = 0.25 at identity 0.9. So retention by (policy, radius, identity) has to be
measured, with the model interpolating between measurements. The table's values are also below 1 at
identity 1 (0.25–0.85), which a useful model must explain.

**Number of reads.** It enters in three ways:

- **Recall:** reads per target = reads × read length × abundance / target length. A community
  richness and evenness turn the read count into a depth per target.
- **Read-level chance:** (read pair, group) chance seeds grow linearly with reads.
- **Sample-level chance:** distinct chance groups grow slower, as reads^~0.5 at 30 bits and ^~0.7
  at 40–50 bits (kfp, 1/64 of reads to all), because sticky leaves take repeat hits. kfp's estimate
  of this curve stops at 261,591 read pairs, so it needs a real 10 M read-pair sample.

Precision is therefore reported at two levels: per read, and per target in a sample of a given size.

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

- **Numeric core: Python and NumPy in Pyodide, in a Web Worker.** Measured under node (Pyodide
  0.27.7), not yet in a browser:
  - Batched NumPy computes the same 7,680 grid values in ~120 ms, against ~165 ms for plain JS. It
    shares work across read lengths, so the two are not strictly like for like.
  - Pure Python loops take ~8.7 s for the same grid, so the model must stay batched.
  - Start-up is ~0.7 s for the runtime and ~0.2 s for NumPy from local files, plus downloading
    about 15 MB uncompressed in a browser. The worker keeps the page usable meanwhile.
  - It shares a language with kfp's workflows and needs no extra toolchain.
- **Measurements: native Python and Rust.** Reuse kfp where it fits: `workflows/tree-index` scripts
  and `crates/core` as a git dependency, where threads and needletail are no obstacle. Write a new
  Rust binary only when a Python script is too slow at scale. Containers via Singularity on HPC.
- **Rust compiled to wasm: not planned.** The model does not need it, and heavy work runs natively.
  If a browser simulation is ever added and NumPy cannot batch a loop, note that `crates/core`
  uses `std::thread`, which does not work on plain `wasm32-unknown-unknown`.
- **UI: JavaScript with `// @ts-check`, no framework, no bundler, Plotly** for hover, log axes and
  heatmaps. Type-checked without a build step; move to TypeScript if the UI grows. Parameters
  live in the URL so a view can be shared.
- **Hosting: GitHub Pages.** Static, with no backend.

## What to measure

The model takes these as inputs. All run outside the browser, and each is summarised into a small
JSON table the page loads.

### Database

| Measurement | Used for | How |
| --- | --- | --- |
| Homologue spectrum: other groups per identity bin, by database size | Precision, close and no-close scenarios | Held-out proteins aligned to the rest, database subsampled (`recall.py holdout`, DIAMOND) |
| Conservation: match-run lengths by identity | Replaces the fitted φ, ρ, ℓ; recall | The same alignments, per aligned position |
| Decoy tail per (k, m), and per-leaf decoy seeds | Noise floor, sticky leaves | Real index, reversed reads (`query.py --decoy`) |
| Group structure per radius: group count and sizes, representative distance, keys per leaf | Recall ceiling, key dispersion, postings | Build outputs, `survey.py`, `wide_clusters` |
| Key retention by (policy, radius, identity), and postings per policy | Effect of grouping and policy | Held-out proteins against keys built each way (kfp's soft intersection test) |
| k-mer spectrum per k: distinct/total, hot tail, holders per k-mer | Index size, lookup cost, chance correction | Counts over a uniform sample, plus kfp's `stream.json` |
| Growth with N: units, distinct k-mers, postings, singletons | Extrapolating to a massive index | Proteins subsampled 1/64 to 1, Good–Turing |
| Length distribution | Short and long breakdown, floors | Counts |
| Composition: global and per protein | Chance baseline | Counts |
| Tree postings against flat postings | Index size if a tree is used | Phase 0 outputs (`gate.tsv`, `measure.json`); one factor per configuration |

### Example queries

Real samples give realistic summaries but have no truth, so use both kinds:

- **Real samples**, about 5–10 across biomes, spanning low to high unknowns and community
  complexity. Summaries only.
- **A simulated grid** of unknowns × complexity × error, with truth, for performance. Spiked real
  backgrounds, as in `workflows/presence-calibration`.

| Axis | Measurement | Used for | How |
| --- | --- | --- | --- |
| Unknowns | Best-hit identity distribution, share with no hit | Mix of identities in recall and precision | Alignment of a read subsample; k-mer hits alone are not enough |
| Community complexity | Richness, evenness, depth per target, read redundancy | Reads per target, depth-limited recall, chance saturation | Profile plus rarefaction of reads |
| Sequence complexity | Read composition, low-complexity share | Local chance rate | Translated frames |
| Error | Read length, base error rate and profile, adapters | Residue error, windows | FASTQ summaries; added errors for the grid |
| Frames | Stop-free frames per read, windows per read | Search space for chance | Translation under kfp's frame policy |
| Noise | Decoy curve per sample | Whether decoy tails transfer across samples, untested so far | Reversed reads of one sample against another's tail |

### Pilot results

Local data only: 44,270 representatives from `data/mini-slice` (not uniform, denser than the
release) and 2,000 reads (150 nt) from fmh-funprofiler's demo. Exploratory, and proteins were
reversed here where kfp reverses reads.

| k | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- |
| Distinct / total k-mers | 0.17 | 0.53 | 0.68 | 0.73 | 0.75 |
| Decoy collisions / composition | 1.2 | 1.4 | 1.9 | 4.7 | 34 |
| kfp's measured correction | | 1.3 | 1.5 | 1.5 | 3.8 |
| Collisions between units / composition | 3.2 | 25 | 306 | 4,195 | 60,106 |

- Decoy rates match kfp at k 6–7. At k 8–9 a few k-mers dominate (the top 1% give 62–87% of hits),
  so this sample size cannot estimate them.
- Above k 5, matches between units are overwhelmingly homology, not chance. This makes the homologue
  spectrum the most important measurement.
- Representative lengths: 5–95% 65–707 aa, E[L²] / E[L]² = 2.6. Length alone therefore explains part
  of the key-count dispersion.
- Stop-free frames per read: mean 1.81, with 13% of reads at none and 28% at three or more. Error
  lowers it to 1.42 at 10%. The check's flat "3 frames" is only a stand-in.
- 6-mer hits per window are 0.30 in the reads' best frame and 0.32 on reversed reads, so at this
  scale they say nothing about known against unknown. Hence the alignment for unknowns.

### Order

1. Homologue spectrum and conservation, from one set of alignments.
2. Decoy tails on a grid of (k, m).
3. Query summaries and the simulated grid.
4. Spectrum, lengths and growth (cheap, and needed early for index size).
5. Group and tree structure, and retention by policy and radius.

Local samples are enough to build and test the pipeline. Values for massive indexes need a uniform
MGnify sample at several sizes, run by the user on HPC.

## Phases

Phases 1–3 are the plan. The rest is a backlog, reordered as questions come up.

**Phase 1: the first page.**

- Batched NumPy chain in a Pyodide worker, with `model_check` ported as a Python test.
- Controls: k, m, read length, identity.
- Plots: recall against identity, and recall against m for several k.
- Readouts: index size at a chosen database size, and chance anchors per read.
- *Done when:* the check's tolerances hold, and slider moves redraw with no visible lag in a browser.

Phase 1 is built. Nothing below it has started.

**Phase 2: analyse a small set in the browser.**

- Load the bundled sample or a FASTA file. Spectrum, lengths, composition, rarefaction and reversed
  decoys, drawn with the model's predictions.
- *Done when:* the 44 k set runs in the browser within seconds per step, and the numbers match the
  native pilot.

**Phase 3: native simulation grid.**

- `measure/` script: index from a local sample, simulated reads at set identities, reversed decoys,
  over a small grid of (k, m). Writes a JSON table.
- Measured points overlaid on the model's curves.
- *Done when:* the simulation agrees with kfp for a matching configuration, using reduced-scale
  numbers from `results-hpc`. This phase also tests the independence assumption for homologues.
  If the model cannot follow the simulation on precision, precision plots show only the grid's
  points: no slider between grid values, unless they are interpolated.

**Phase 4: score, noise floor and precision.** Hit-count distribution, per-key bits, decoy tail
tables, an information bit floor slider, a number of reads control with depth per target,
precision per read and per target, the PR curve and AUPRC under one scenario.

**Phase 5: index selection, scenarios and scale.** A selector for the index, with each index
supplying its target lengths, group sizes and homologue spectrum (mini-slice and MGnify at first, a
loaded FASTA from phase 2). Homologue spectra and presets, a database size axis, comparison of small
and massive indexes, and a breakdown of recall, precision and postings by target length.

**Phase 6: grouping and key policy.** Controls for radius and policy. Recall, precision and index
size respond to both, from measured retention by (policy, radius, identity). Relatives within the
radius count as correct. *Done when:* the model reproduces kfp's matched comparisons, such as
`tree:0.34`, `tree:0.57` and `tree:0.83` at composite 0.25, and composite 0 against 0.25.

**Phase 7: SNP and overlap panel.** The miss probabilities above, with a key-placement diagram.

**Phase 8: optimiser.** Heatmaps of AUPRC against index size over k × m. The best parameters per
scenario, under an index size budget.

**Phase 9: combination k-mers.** Added to the model and the simulator.

**Measurement workflow (runs alongside phases 4–5).** `measure/` grows into a Nextflow workflow that
produces the JSON tables in [What to measure](#what-to-measure), shipped for the user to run.
It includes the [scale profile](#scaling-to-full-size). Anything at full MGnify scale runs on HPC
by the user.

## Progress log

- **2026-11-04: phase 1.** `web/` runs the model in a Pyodide worker, with k, sparsity, read length,
  identity, database size, representative share and paired reads as controls, held in the URL.
  - `checks/test_model.py` passes: chain against Monte Carlo within 0.005, fit on m 8 gives RMSE
    0.010 and held-out m 16 gives 0.024, chance within 2× of kfp, postings within 30%.
  - In the desktop app's browser pane: start-up 1.3 s (5.7 s on another load), compute 23–105 ms.
    During a simulated 2 s drag the page drew 19 times, about 10 per second, with the longest
    main-thread gap at 197 ms. Whether this counts as no visible lag is for the user to judge.

## Licence

GPL-3.0-or-later.
