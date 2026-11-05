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
| Noise floor tail (chance against score bar) | Independent hits fall ~1,000× per 10 bits. Decoys fall 10–12× (2.5–5× on real tiers at 40–70 bits; progress log, 2026-11-05) | **No: fit it to decoys** |
| Precision among homologues | Measured shared k-mer counts predict it at any sparsity and hit threshold, on average over hashes. Best-group assignment and growth to massive indexes are not yet predicted | **Yes per index, from a small measured table; see below** |
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
  so this sample size cannot estimate them. These figures reverse the whole set; see the phase 2
  entry in the progress log for half-set decoys.
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

Phases 1–3 and 3b are built, phase 3c is partly done, phase 4 is mostly done, phase 6 has tables and a panel, precision is computed on demand at k 1–31, and phases 7 and 9 are built (below). Phases 5 and 8 have not started; both need results at scale.

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

**Phase 3b: correlated hits.** Added after phase 3, before phase 4. Hits on a read's true group and
its relatives share two things, so independence holds only given both:

1. **A rate per read,** λ ~ Gamma(a, 1/a), with identity to anything = identity^λ. In the model and
   page.
2. **Hash sampling,** which is shared: a k-mer that two groups both hold is kept or dropped for both.
   Needed for reads that miss their true group.
3. **What the unaligned hits are.** 53–84% of other-group hits fall outside each query's 50 aligned
   candidates. Align more, and align a sample of those hits directly.
4. **Fit on another set.** Repeat on `tree-index-slice`, to find out whether a default works or each
   index needs its own fit.

**Phase 3c: extend the measurements to the wider ranges.** The page's ranges are ahead of what has
been measured:

- **Nucleotide simulation.** Nucleotide mode is unvalidated. It needs relatives at the nucleotide level
  (gene sequences, which `measure/` does not have yet), canonical k-mers, composition from the set,
  and a mapping between amino-acid and nucleotide identity. Codon structure matters: third positions
  vary most, so matches are periodic, which the chain does not model. Fit its own conservation.
- **Larger k and sparsity in the grid.** The grid stops at k 13 and 1 in 128. Beyond k 13, base-21
  codes overflow an int64, so hash k-mers (128-bit for nucleotide k above 31). Sparse settings need
  more reads per bin, since few k-mers survive.
- **Browser analysis for other alphabets.** It is protein only, rejects DNA and stops at k 13. Encode by
  alphabet, with k limited by 63 bits / log2(letters) for nucleotide (31) and hashes beyond that.
- **The remaining reduced alphabets** through `sim_grid.py --alphabet`, to test the 1.6× scale.

**Phase 4: score, noise floor and precision, from shared k-mer tables.**

- `measure/` writes, per index and k, a table of M by (identity bin, truth, relative identity), for
  reads and decoys, on 1,000 or more queries. The page computes recall, others and precision from it for
  any m and hit threshold, in milliseconds. M should also carry bit bands for kfp's per-key bits.
- Controls: information bit floor, truth level (exact to x ≥ 0.5) and identity mix, including reads with
  no relative. PR curve and AUPRC by sweeping the floor. A range for the hash draw.
- A number of reads control with depth per target, and per-target precision. Untested.
- Best-group assignment from a stored panel of reads, or from the grid at set values.
- *Done when:* page values match a held-out simulation at settings not used for its tables.
- The analytic model stays for what has no table: other k and quick exploration. Tables win where they exist.
- **Not done yet:** tables for other alphabets and policies, per-target checks beyond depth 6 or with a
  floor, kfp's sampling of rare keys, and groups larger than one representative.

**Phase 5: index selection, scenarios and scale.** A selector for the index, with each index
supplying its target lengths, group sizes and homologue spectrum (mini-slice and MGnify at first, a
loaded FASTA from phase 2). Homologue spectra and presets, a database size axis, comparison of small
and massive indexes, and a breakdown of recall, precision and postings by target length.

**Phase 6: grouping and key policy.** Controls for radius and policy. Recall, precision and index
size respond to both, from measured retention by (policy, radius, identity). Relatives within the
radius count as correct. *Done when:* the model reproduces kfp's matched comparisons, such as
`tree:0.34`, `tree:0.57` and `tree:0.83` at composite 0.25, and composite 0 against 0.25.

Run `measure/precision_tables.py` on keys built kfp's way (group radius, then soft intersection or
frequency ranking, with kfp's key budgets), not only on single representatives. This also tests whether
those groups explain kfp's heavier chance tail: at 40 bits and k 5 the current tables are 4–25× below
kfp's decoys, probably because of sticky leaves.

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

- **2026-11-04: phase 2.** The page analyses a bundled mini-slice sample (10 k or 44 k representatives,
  `measure/make_samples.py`) or a FASTA file in the browser: lengths, composition, spectrum, decoys,
  matches between halves and rarefaction, each beside what independent residues would give.
  - The 44 k set takes 14 s in the desktop app's browser pane, at most 3.4 s per step. The 10 k set
    takes 5 s. During a run, the model's sliders wait for the current step (3.5 s when timed).
  - `checks/test_analyse.py` passes: distinct shares and matches between halves agree with the pilot,
    and random sequences match the independent-residue values.
  - Decoys changed from a whole set reversed to one half reversed against the other, because a set
    matches its own palindromes (1.18 at k 7 on random sequences, 1.10 after, 1.00 expected, with
    the rest probably noise). Against the pilot: 1.23, 1.40, 1.90, 4.29, 28.4 at k 5–9. So the
    self matches explain little, and k 8–9 stay far above kfp's correction in a set this small.
  - Not included: a sampled index and simulated reads, listed under "Analysing a small set in the
    browser". Simulated reads need realistic divergence to say anything, so they stay with phase 3.

- **2026-11-04: phase 3.** `measure/sim_grid.py` (run from kfp's checkout, for its aligner) builds an
  index of 38,270 MGnify90 representatives from the mini-slice, with 22,500 read pairs (2 × 50 aa) from
  2,250 proteins, 250 per identity bin, at k 5–11 and m 1–128. It takes about 2 minutes on two
  threads with a 1.4 GB peak. Points appear on the page's recall plots. `checks/test_grid.py` passes.
  - **Against kfp, at k 9:**

    | | Simulation | kfp |
    | --- | --- | --- |
    | Chance anchors per pair, 3 frames, m 8 | 0.013 | 0.014 |
    | Pair recall at 0.9, m 8 | 0.99 | 0.97 |
    | Pair recall at 0.9, m 16 | 0.94 | 0.88 |

    Recall also runs above kfp's table at lower identity and greater sparsity, by up to 0.15 (k 9, m
    16, identity 0.8: 0.80 against 0.65). Dense settings agree once kfp's 0.96 ceiling is applied. The
    simulation has no verification step and measures identity to the representative itself, not to a
    member of a larger group, which are likely causes, but neither has been tested. So the "done when" condition holds for
    chance, and only partly for recall. **Correction (phase 3b):** this entry first reported 16.0
    groups per pair at identity ≥ 0.5 against kfp's 16.1 verified. That agreement came from aligning
    only 50 candidates per query. With 200 it is 30.2, so it does not support the simulation. The two
    also count different things (whole-protein identity here, verified read windows in kfp).
  - **Recall model, not refitted:** RMSE 0.057 over the whole grid with the ceiling removed, against
    0.128 for independent mutations. Bias is within 0.005 at identity ≥ 0.8, but the model is too
    low by 0.096 near 0.55 and 0.07 near 0.65.
  - **Chance:** decoys over composition at m 8 are 1.42, 1.77, 2.02 and 4.00 at k 6–9, against
    corrections of 1.3, 1.5, 1.4 and 3.6, so an independent sample supports them. They rise with
    sparsity and k. At k 9, m 8, decoy hits are about 0.01% as common as other-group hits on this set, so homology
    dominates at this k.
  - **Independence fails.** At relatives of identity ≥ 0.5, a read hits a relative at 0.60 if its
    true group is anchored and 0.14 if not (k 9, m 8), and 0.44 against 0.06 at m 32. Reads in
    conserved regions hit everything, and reads elsewhere hit little. Pooled precision is a ratio of
    expected counts, so this does not bias it, but it does matter for how many reads are ambiguous.
  - **Precision fails.** Hit rates on low-identity relatives are far above the model's (0.16
    against 0.013 at relative identity 0.4–0.5, k 9, m 8, when the true group is anchored), since a
    whole-protein identity hides shared conserved domains. And 53–84% of other-group hits fall on
    groups outside each query's 50 aligned candidates, or below 0.3 cover. Without any filter,
    simulated precision is 0.01–0.05 per pair and the model's 0.03–0.12. So phase 4 reads precision
    from the grid, as this plan's fallback says.
  - The mini-slice is denser than the release, so none of these precision values transfer to it.

- **2026-11-04: phase 3b.** Grids at 200 aligned candidates per query: `web/data/sim-grid.json`
  (mini-slice) and `sim-grid-slice.json` (`tree-index-slice`, 1/16 of its units, 33,400 groups).
  `measure/fit_rate.py` fits them. All three checks pass. Figures in the phase 3 entry used 50
  candidates and no read rate, and this entry supersedes those on precision.
  - **Read rate:** λ ~ Gamma(a, 1/a), fitted to true-group recall at k 6–9, m 8. Both sets give a = 4,
    on a coarse grid of shapes, and each set's recall is unchanged under the other's. Both sets are
    MGnify samples, though selected differently, so this is not yet evidence for other databases.
    Recall RMSE on the rest of the grid: 0.044 and 0.043, against 0.064 without the rate. Model bias
    at identity 0.6–0.85 is −0.03 to −0.05, and within 0.005 at 0.85 and above. The fit to kfp's table
    stays at RMSE 0.029. The page uses it, at 23–60 ms per update.
  - **Hits on aligned relatives per pair**, predicted with no further fitting: 35.5 against 36.2, 27.0
    against 26.4, 14.6 against 14.7 and 21.3 against 20.3 on the mini-slice, at (k, m) = (7, 8), (9, 8),
    (9, 32), (11, 8). On the slice: 30.9 against 31.1, 22.2 against 20.7, 10.9 against 10.3, 16.5
    against 14.1.
  - **Unaligned hits are real relatives.** Even with 200 candidates, 20–50% of other-group hits are
    outside the aligned set, by set and setting. In samples at k 7 and k 9, m 8: 88–97% score above the 99th
    percentile of decoy hits, and 75–86% are whole-protein relatives (identity and cover ≥ 0.3) that
    were simply not among the candidates, and 5–9% were candidates below 0.3 cover.
  - **Precision:** counting those hits as measured, predicted precision is within 20% of simulated
    in `checks/test_grid.py` (for example 0.019 against 0.020 at k 9, m 8). Without them it is 1.1–1.7×
    too high. So the model needs a complete homologue spectrum. Unfiltered precision is still only
    0.01–0.04, and these sets are denser than the release.
  - **Shared hash, partly solved:** `model.hit_both` handles a k-mer that two groups both hold, and
    matches Monte Carlo. One share parameter does not fit: the best share rises from 0 at relative
    identity below 0.6 to 1 at 0.8 and above. Those shares, fitted at k 9, m 8 on the mini-slice, reduce
    the error of P(relative hit | true group missed) elsewhere: RMSE 0.09–0.14 from 0.16–0.21 on the
    mini-slice, 0.10–0.12 from 0.16–0.18 on the slice (0.015 and 0.025 at m 32). This affects
    how many reads are ambiguous, not pooled precision. Not on the page.
  - **Next, for phase 4:** make the spectrum complete, by aligning more candidates or estimating the
    count from hits, and read a and the spectrum per index. Local identity (per read window) may
    replace whole-protein identity for relatives, which is untested.

- **2026-11-04: wider ranges and alphabets.** The page now covers k 1–31 (1–51 for nucleotide),
  sparsity up to 1 in 8,192 (2¹³, the closest to 9,000), nucleotide mode and 19 reduced protein alphabets
  (kfp's classes), and shows chance bits per k-mer. The chain matches Monte Carlo at k 1, 31 and 51.
  - **Chance:** class frequencies predict decoys. Decoys over composition at m 8 are 1.0–1.5 for
    murphy10 at k 7–11, and 1.05–1.6 for seb8 and dayhoff at k 8–11. It rises above that at k 13 (4.3
    for murphy10).
  - **Recall:** BLOSUM62 understates substitutions staying within a class. Best fits are 0.38 against
    0.23 (murphy10) and 0.52 against 0.33 (dayhoff). So the model scales within-class shares by 1.6.
    Recall RMSE: 0.033 and 0.031, from 0.071 and 0.097. seb8, not used for the scale, gives 0.030, from
    0.079. The other 16 alphabets are untested. `sim_grid.py --alphabet` runs any of them.
  - **Reduced alphabets at matched chance:** on the mini-slice at m 8, protein k 7 (28.5 bits) has 0.54
    decoys per pair and recall 0.36 / 0.80 / 0.97 at identity 0.55 / 0.72 / 0.82. murphy10 k 11 (30.0 bits)
    has 0.14 and 0.27 / 0.75 / 0.96, and dayhoff k 13 (27.6 bits) has 0.97 and 0.34 / 0.75 / 0.96. This
    shows no clear gain, in line with kfp's result for murphy10.

- **2026-11-04: what precision needs.** `measure/precision_tables.py`, on the mini-slice and
  `tree-index-slice` (4,050 read pairs each, protein reads, k 7, 9, 11, one set of representatives).
  - **The sufficient statistic is M,** the distinct k-mers a read pair shares with a group at full
    density. Each is kept at 1/m, so hits ~ Binomial(M, 1/m). A table of M by (identity bin, truth,
    relative identity) from one dense pass gives recall, others and precision at every m and hit
    threshold, with no alignment beyond labels. Against real hash sampling averaged over 8 salts, at m
    8 and 32: recall within 0.006, and others and precision within 10% where counts are common, up to
    about 17% in rare settings, where 8 salts is itself noisy.
  - **One index is one hash draw.** Sd across salts: precision ±5% at m 8, h 1; ±11–40% at m 32, h 3
    (0.093 ± 0.037 on the mini-slice at k 9). A few popular k-mers cause this, and the page should
    show it as a range.
  - **A score does not work as the link.** At the same ungapped score, hit rates differ 2–3× between
    relatives (k 9, m 8, S 150–200: 0.28 at x < 0.5, 0.67 at x ≥ 0.7). The tables built on it were
    20–30% off, and about 35% on another set. Removed.
  - **Truth level and rule matter most.** k 9, m 8, precision against the true group only, or also
    relatives at x ≥ 0.9, 0.7, 0.5:

    | Rule | Set | Recall | Exact | 0.9 | 0.7 | 0.5 |
    | --- | --- | --- | --- | --- | --- | --- |
    | Any anchor | mini | 0.80 | 0.02 | 0.12 | 0.41 | 0.80 |
    | ≥ 5 hits | mini | 0.44 | 0.06 | 0.43 | 0.87 | 0.98 |
    | Best score | mini | 0.54 | 0.18 | 0.72 | 0.93 | 0.99 |
    | Any anchor | slice | 0.78 | 0.03 | 0.05 | 0.24 | 0.72 |
    | ≥ 5 hits | slice | 0.41 | 0.16 | 0.36 | 0.77 | 0.99 |
    | Best score | slice | 0.57 | 0.44 | 0.62 | 0.84 | 0.97 |

    A read-level score bar hardly helps: S ≥ 100 leaves others per pair almost unchanged (39.7 to 38.4),
    since nearly all anchors are on homologues. It removes chance, which is rare at k ≥ 9.
  - **Best-group assignment is not predicted by M.** Thinning each group independently gives
    precision 0.29 against 0.14 (k 9, m 8), and 0.17 against 0.07 at m 32, because k-mers shared across
    groups are kept or dropped together. It needs the read × k-mer × group incidence, for example
    a stored panel of reads thinned by salted hash.
  - **Reads needed:** relative sd of precision is 0.21 at 100 queries, 0.15 at 200, 0.11 at 400 (k 7). So
    use about 1,000 or more.
  - **Scale is not answered.** Removing groups at random here gives others ∝ groups (exponents
    0.98–1.03), but that is true by construction. Real growth needs the scale profile's uniform samples.
  - **Also not covered:** reads with no relative (8–10% of queries here, more in real samples), translated
    frames and read errors, per-target precision across reads, per-key bits in the floor (kfp's bits vary
    by key, so M would need bit bands), and kfp's own key policies and verification.

- **2026-11-04: phase 4, tables.** `measure/precision_tables.py` writes M tables for k 5–9, 11, 13
  (`web/data/tables-mini-slice.json`, `tables-slice.json`, about 600 KB each). The page has a precision panel:
  index, information bit floor, truth level, identity mix, reads with no relative, read pairs and targets,
  with precision, recall, AUPRC and a PR curve, at any sparsity. About 20 ms per update in the browser, with
  1.0 s start-up. `checks/test_tables.py` passes.
  - **Held-out check** (mini-slice, different queries, 9,000 read pairs, k 7, 9, 11, m 1–128, against 8
    hash salts): recall RMSE 0.007; others 14% and precision 14% (log-ratio RMSE, settings with at least 1
    other per pair); precision at relative identity ≥ 0.7 is 5%, against one hash draw. The tables run
    slightly low on others (for example 35.7 against 37.7), so precision reads slightly high. The slice
    has no held-out run yet.
  - **The first test was unfair:** unsalted hashing is the same draw in every run, and it runs high here (22
    others per pair at k 9, m 32, against a mean of 17). Comparing means over salts removed most of the gap,
    with limits unchanged. Index-specific draws do matter, hence the range on the page.
  - **Per target, depth 6:** reads of one target share conservation, so independent reads gave recall 0.80
    against 0.66. Using each query's own M reduces RMSE to 0.080, right at the check's limit. It stays 0.09–0.14
    too high at m 32, probably since reads of one target share k-mers and hash decisions. False groups:
    log-ratio RMSE 0.21. Greater depth is untested and likely worse. The chance term's growth exponent is kfp's.
  - The floor here used one bits value per k-mer. Replaced by the next entry.
  - At k 5–6 relatives are mostly not aligned, and count as wrong for "or a relative" truth levels.

- **2026-11-04: real floors and best-group assignment.** The tables now hold each read's real score across
  floors none, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200 and m 1 to 8,192, over 8 salts, and the best-group
  rule at margins 0, 3 (kfp's default) and 10 bits. Scores are kfp's evidence: key bits = −log2 q0 − log2 g
  (g here the groups holding the k-mer, in kfp the key's gains), less log2(windows in all 6 frames) per hit
  after the first. The page has a discrete floor slider and plots across floors for every k. About 30 ms per
  update, 1.1 s start-up, tables 2.7 MB per index. `checks/test_tables.py` passes.
  - **Calibration, nothing fitted.** At k 5: 9.1 + 7.7 bits per hit, against kfp's fitted 9 + 8. Mean key bits
    on the slice: k 7, 29.0 against 29.1 in kfp's index; k 11, 45.7 against 45.4. kfp's sample keys
    rarest-first and on composites, so these agree on scale, not sampler.
  - **Decoys at k 5 against kfp:** at 30 bits, 0.03–0.16 per pair (m 16 to 8), which brackets kfp's 0.089. At 40
    bits, 0.0003–0.0018 against kfp's 0.0074, so **4–25× too low**. kfp's tail is heavier, probably from sticky
    leaves: its groups hold keys in proportion to √members, and these are single representatives.
  - **Reads over a high bar:** kfp at 90.5 bits has 0.77 / 0.46 / 0.29 / 0.13 / 0.02 at identity 1.0 / 0.95 / 0.9 /
    0.8 / 0.7. At m 8 the true group at 80 bits has 0.68 / 0.62 / 0.45 / 0.19 / 0.04, and at 120 bits 0.20 /
    0.16 / 0.10 / 0.03 / 0.01, which brackets kfp except at identity 1.0. At m 16 it is far lower, though kfp has
    a key per ~12 residues, so a kfp index behaves like a denser hash-sampled one.
  - **Held-out** (different queries, means over salts, k 5, 7, 9, 11, m 1–128, all floors). Every group over the
    floor: recall RMSE 0.007, others 12%, precision 12%, precision at x ≥ 0.7 is 5%. Best group within 3 bits:
    recall 0.005, others 17%, precision 14%, precision at x ≥ 0.7 is 3%. Tables read slightly high on precision.
  - **Per target at depth 6:** called RMSE 0.037. False groups are too high at sparse settings and small k (k 5, m 32:
    298 against 146), though close at k 9, m 8 (61 against 62). Tested with no floor only, against one hash draw.
  - **What the values show, on the mini-slice** (k 9, m 8, natural mix, 5% no relative). Every group: precision
    0.63 at x ≥ 0.7, 48 others per read. Best within 3 bits: 0.96 at x ≥ 0.7, or 0.11 against the true group
    alone, recall 0.85. The same on the slice: 0.48 against the true group alone.
  - **Low floors:** floors below a key's bits change nothing. At k 9 floors up to 30 are identical (5–95% of keys
    hold 34–41 bits). They matter at k 5 (20 bits: precision at x ≥ 0.7 rises from 0.08 to 0.52, recall 0.98
    to 0.96) and slightly at k 6. Floors under about 15 bits would need k 4 or below, which has no table.

- **2026-11-05: kfp's real indexes on the page.** `measure/benchmark_summary.py` reduces kfp's `tree-index-benchmark` run (1-in-100 subset, 2,000 strains, 9,191 read pairs, no background) to
  `web/data/kfp-benchmark.json` (83 KB). A new panel plots recall by index size, recall at each precision, recall by strain identity and the decoy tail, for three flat tiers and three tree tiers. These are measurements, not the model. `checks/test_benchmark.py` passes: 36 gate rows lie within the table's precision grid (widest bracket 0.004 recall), and the size fit is within 6% on each tier.
  - **Index size.** Tree tiers fit 1.9 GB plus 19.6 bytes per posting, which supports the plan's 19–20 B and adds a fixed term the page's index size lacks. The page still uses 6 B (the anchor design), with no fixed term. The composition of the fixed term is unread: the indexes are not in the results.
  - **Chance grows linearly in postings, at real keys.** Decoy leaves per pair at ≥ 30 bits, less slim's, are 1.10 (standard) and 5.81 (large), a ratio of 5.3 for 5.4× the k 8 layer's postings. The slice's k 8 table at m 8, scaled by postings and 3 frames, predicts 2.5 and 13.7: **2.3× high on both**, plausibly because these keys are the rarest per window and the table's are hash-sampled; not tested. Groups count once there and leaves here, which should push the other way.
  - **The tail is heavier than this plan's estimate.** Decoys fall 2.5–5× per 10 bits from 40 to 70 bits (slim 360 to 3 leaves, large 1,764 to 115), against the 10–12× in the Verdict and "~10× per 10 bits" under Noise floor. The steps are not geometric: 7–32× from 30 to 40 bits, probably where single k 8 and k 7 hits end. Counts above 80 bits are 20 leaves or fewer.
  - **Sticky leaves are not visible at this read count.** Among aligned decoys, the top 1% of leaves hold 5% of (leaf, read) rows, on 56,000–197,000 rows, and at most 26 for one leaf. That does not support stickiness as the cause of the k 5 gap in the last entry, but 9,191 read pairs is small and this only sees decoys that reached alignment. The k 5 gap itself is untested here: no tier uses k 5, and the k 8 tables are 1–7× high at 40 bits, on few counts.
  - **Not usable from this run:** per-(k, m) tails, since layers are pooled; retention by policy and radius, since keys and indexes are not in the results; anything about growth, with one index size; precision against background.

- **2026-11-05: phase 6, grouping and key policy.** `measure/policy_tables.py` holds out 4,000 proteins, then builds keys with kfp's own `layer_groups`, `window_kmers`, `on_reps` and `allocate` on the other 274,554 mini-slice proteins, for 32 configurations. 3,000 read pairs and reversed decoys (100 queries per identity bin, 3 pairs each) are scored with kfp's evidence: in frame, no errors, no base index, no verification. It takes about 25 minutes on two threads, 10.6 GB peak. The page's new panel compares policy, radius, window, budget and k at any floor (`web/data/policy-mini-slice.json`, 114 KB). `checks/test_policy.py` passes.
  - **Phase's "done when": met for ordering, not for size of effect.**
    - Retention against kfp's table (32 keys, groups of at least 8): RMSE 0.087, and frequency > share ≥ 0.25 > union at every bin from 0.7 up. Frequency over share is 1.27–1.41× at 0.7–0.9, against kfp's 1.4–1.5×. Values run 0.15–0.20 high at 0.8–0.85, on 26–75 queries per bin.
    - kfp's matched comparisons, 10 configurations at window 8: rank correlation 0.66 for recall and 0.75 for postings. Effects are larger here: composite 0.75 falls by 0.20–0.26 in recall per target, against kfp's 0.08–0.12, as expected with no base index or neighbour rule, but that has not been tested. Postings relative to composite 0.25 are within 0.86–1.13× of kfp's among policies, but 1.34× for `units` and 1.24× for `tree:0.34`. kfp's are on the quarter sample with Fitch sharing and these are raw on a denser set, which may explain it.
  - **Grouped keys explain most of the heavier chance tail.** Decoys per posting at k 5, window 8: 40 bits, 1.6 × 10⁻⁹ with no groups (as the hash-sampled tables), 13 × 10⁻⁹ on `tree:0.57` composites and 4 × 10⁻⁹ with `sqrt:14` (8 decoys). The rise is 8.5× at k 5, 5.8× at k 6 and 2.4× at k 7 at 40 bits, and 2–7× at 30 bits. On `sqrt:14` at k 5, decoys per pair are 0.118 at 30 bits and 0.0027 at 40, against kfp's 0.089 and 0.0074: now within 3×, from 4–25×. It is not a few sticky groups: at 30–40 bits the top 1% of groups hold 4–6% of decoys where counts are large. More keys per group, so more multi-hit chance, would fit, but is untested. In absolute terms grouping costs little: 6× fewer postings at about the same decoys per pair at 40 bits (0.045 against 0.032, k 5).
  - **Frequency ranking is not a clear gain at matched chance, the first end-to-end test of kfp's proposal.** At `sqrt:14`, k 6, at the same floor (40 bits) it adds recall at low identity (per target 0.35 against 0.23 at 0.7–0.75, 0.69 against 0.59 at 0.8–0.85), but has 3.6× the decoys (29 against 8) and lower precision (0.45 against 0.56 at x ≥ 0.7). At matched decoys (frequency at 40 bits, share at 30, 29 against 28): per target 0.35 / 0.55 / 0.69 / 0.83 / 0.90 against 0.28 / 0.62 / 0.69 / 0.83 / 0.95 at 0.7 / 0.75 / 0.8 / 0.85 / 0.9, with precision 0.45 against 0.42. Differences are within noise (about ±0.05 per bin). The extra hits appear to come with extra chance.
  - **Groups against singles.** k 6, window 8: `tree:0.57` composite 0.25 uses 0.15× the postings of one key set per leaf, for per-target recall of 0.92 / 0.96 / 0.93 against 0.87 / 0.96 / 1.00 at 0.7 / 0.75 / 0.8 at 40 bits, and precision at x ≥ 0.7 of 0.21 against 0.18. Radius changes postings by 0.91–1.41× with small recall changes, in line with kfp's "grouping matters less than the window".
  - **Limits.** One hash and read draw, on a set denser than the release. Decoys above 40 bits rest on few counts. Identities of relatives come from the top 100 candidates per query, others count as wrong for "or a relative". The page's other panels (anchor recall, precision) do not yet take radius or policy.

- **2026-11-05: on-demand precision, timing test.** A throwaway script, not on the page: the steps of `precision_tables.py` for one (k, m) in Pyodide in a worker, in the desktop app's browser pane, on the 44 k sample (42,770 groups after holding out 1,500 queries; 4,500 read pairs and their decoys, 8 salts, 11 floors, every-group and best-group rules, per-target). One run, on this laptop. Relative labels and table output are not included, and truth bands are stand-ins, so this times the work, not the values.

  | k | m | Read pairs | Rows | Index, s | Find hits, s | Score, s | Total, s |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 13 | 8 | 4,500 | 110 k | 19.1 | 3.5 | 1.8 | 24.4 |
  | 11 | 8 | 4,500 | 160 k | 20.8 | 3.1 | 2.2 | 26.0 |
  | 9 | 8 | 4,500 | 266 k | 19.5 | 2.9 | 3.1 | 25.5 |
  | 9 | 1 | 4,500 | 266 k | 17.3 | 2.8 | 3.8 | 23.9 |
  | 7 | 8 | 4,500 | 545 k | 6.7 | 1.3 | 0.9 | 8.9 |
  | 7 | 1 | 4,500 | 545 k | 5.5 | 1.3 | 1.4 | 8.2 |
  | 6 | 8 | 4,500 | 1.1 M | 5.2 | 1.3 | 1.4 | 7.9 |
  | 5 | 8 | 4,500 | 5.6 M | 4.8 | 1.3 | 4.8 | 10.8 |
  | 5 | 8 | 500 | 619 k | 5.3 | 0.1 | 0.5 | 5.9 |
  | 4 | 8 | 100 | 1.1 M | 4.2 | 0.2 | 0.8 | 5.1 |

  - **On-demand is feasible at k 5–13 with no cap on queries:** 8–26 s the first time, then about 1–5 s for a new m and about 2–7 s for a new read length, once the index is held in memory. Start-up is about 4 s. The wasm heap reached 1.36 GB.
  - **The slow step is not the one expected.** Low k was the worry, but k 5 takes 11 s. Building the index takes 17–21 s at k ≥ 9 and about 5 s at k ≤ 7, for reasons not traced. It runs once per (k, alphabet), so it is the step to speed up, and caching it may help more than caching results.
  - **k 4 needs a cap.** 100 read pairs give 1.1 M rows, so 4,500 would give about 50 M. About 500 pairs should match k 5's cost; that is an extrapolation. k 1–3 are untested and will be heavier.

- **2026-11-05: precision on demand.** Any protein or reduced-alphabet setting at k 5–13 now gets precision plots, at any read length, paired or not, on either index. Bundled tables still answer protein read pairs of 50 aa at their k, instantly. Everything else goes to `web/live.py` in its own Pyodide worker (`web/live.js`), which computes one sampling level for one k and caches the result in IndexedDB, keyed by version and settings, so a setting is computed once per browser. `measure/export_live.py` writes the only pre-computed input, `web/data/live-*.npz`: sequences, 1,500 queries with bins and true groups, and aligned identities for every (query, group) sharing a 7-mer (7–8 MB per index, about 4 minutes natively). Results have the tables' shape, so `model.compute_precision` reads them unchanged.
  - **Against the tables,** on different queries (`checks/test_live.py`; mini-slice, 6 settings of k and m, all floors): recall RMSE 0.005, other groups log-ratio RMSE 0.128, running 8–14% below the tables at no floor. Later checks confirm longer reads raise recall, single reads lower it, and a stored result gives the same panel.
  - **Times in the desktop app's browser pane, one run:** about 7 s for a new k at k 10 (the index step was 20 s in the timing test; a stable sort and no second sort cut it), 7–17 s for murphy10 k 11 and slice k 12, about 1 s for a new m and 2–3 s for a new read length. Start-up adds about 4 s, and the first use downloads 7–8 MB.
  - **Not covered, and the panel says so:** nucleotide mode (no nucleotide relatives in the bundled sets) and k outside 5–13. k above 13 needs hashed k-mers and k 4 needs capped reads. Database size and representative share do not change this panel, which describes the bundled set: that needs the scale profile.
  - **Limits.** Read count is fixed at 3 read pairs per query, so a read length above half a query's length drops that query, and very long reads shift the bins. Reduced alphabets have no check of values, only that they run and stay in range. Relatives are found by protein 7-mers, so on reduced alphabets and at k 5–6 more hits have unknown identity and count as wrong for "or a relative". Plots across k show only the k computed so far at these settings.

- **2026-11-05: recall per target, by coverage.** The precision panel now plots recall per target, at 1×, 2×, 5×, 10×, 20×, 50× and 100×: against precision per read, by floor, by coverage for each k computed, and by identity. A coverage control replaces the read count, which now follows from coverage and targets. The panel's plots no longer use the bundled tables, because those hold too little to give recall per target, so every setting is computed on demand and the first view takes the download and computation. The tables stay as a check.
  - **Method (`live.target_called`).** Three random read pairs per query cannot give recall at high coverage: a target with none passing would stay undetectable at any depth. So reads are tiled across each target, every half read length. Within one hash salt, a target's passing share r comes from its tiles, and it is called with probability 1 − exp(−n r), with n = coverage × its own length / (mates × read length). Only hits on the true group are needed, so this is cheap.
  - **Check (`checks/test_live.py`).** Against simulated random reads at the same coverage, salts and floors, on 300 targets and 8 settings from 1× to 100×: RMSE 0.039 over identity bins. Means per setting are within 0.02 except k 9, 1 in 32, 10× at 40 bits (0.566 against 0.599) and k 7, 1 in 64, 50× at 40 bits (0.358 against 0.408). So it runs a little low at sparse settings over a floor, possibly because tiled pairs have one mate gap where real pairs vary.
  - **Limits.** Recall counts every read over the floor on the true group, so "A read keeps" does not change it; a best-group rule per target is not measured. Abundance is even, with no errors or translation. Precision per target still uses read-level hit rates, is untested above 6 read pairs per target, and probably reads high at great depth, since other groups reached from elsewhere on a target are not counted, so the plots pair recall per target with precision per read.

- **2026-11-05: precision at every k on the slider, k 1–31.**
  - **k above 14 is hashed.** The base-21 code fits 64 bits to k 14, and above that it wraps, which `level()` then mixes. So the code is unchanged to k 14 and a hash beyond it, and key bits now come from a running sum along the sequence, not from decoding the k-mer. Across the join, recall per target is 0.964 at k 14 and 0.959 at k 15 (m 1, no floor); k 20 gives 0.940 and k 31 0.922.
  - **Short k-mers are capped by hits, not by k.** Read pairs are cut, evenly across identity bins, to stay within 15 M k-mer hits, measured from a sample of reads. This also protects reduced alphabets: dayhoff at k 5 has a k 2–3 protein's few k-mers, and is measured on 40 read pairs. On the mini-slice, protein: k 4 uses 870 read pairs, k 3 uses 80, k 2 and k 1 use 10. The panel shows the count. Recall per target still uses every query, since it only needs the true group.
  - **"No floor" now means any hit.** It was a score of at least 0, which dropped multi-hit reads on keys with no bits, so recall at k 1–3 read near 0. This changes results only where key bits fall below the charge.
  - **Times in the browser pane, while the pane was hidden, one run each:** k 1, 9.8 s; k 3, 7.7 s; k 4, 12.7 s; dayhoff k 5, 5.5 s; k 14, 26 s; k 20, 30.5 s. High k is the slowest, and probably sorting mostly distinct k-mers is why, but that has not been profiled. Wasm heap was not re-measured with these changes.
  - **Limits.** At k 1–3 precision reads 0.000, and with 10–80 read pairs it is noise beyond that: nearly every group is hit. At high k with sparse sampling, few hits survive. Nucleotide mode is still unavailable.

- **2026-11-05: phase 7, SNP and overlap panel.** New panel: key placement on a read (hash sampled, evenly spaced or window minimizers, at the page's k, m, read length and pairing), drawn with the fewest substitutions that remove every key, the chance d uniformly placed substitutions do so, for that placement and averaged over 32, and the chance a read holds no whole key. SNP and read error rates give a letter identity, and anchor recall both under uniform changes and from the model's chain.
  - **Method.** The fewest substitutions are a greedy hitting set over key intervals, optimal on a line. The removal curve counts sets of positions exactly, by a recurrence left to right, in `model.removal_curve`. A read's chance of no key follows from spacing: (1 − 1/m)^W sampled, max(1 − W/m, 0) evenly spaced, and (m − W)(m − W + 1) / (m(m + 1)) for minimizers with gaps taken as uniform, W being the read's windows.
  - **Checks (`checks/test_snp.py`).** Hitting set and curve match every position set on 200 random small reads, within 10⁻⁹. No-key chance is within 0.03 of simulated reads on 8 settings (the largest gap 0.017). Recall under uniform changes is within 0.02 of simulation.
  - **Limits.** Positions are read letters; a substitution is taken to break exactly the keys overlapping it, so a change that happens to create another key is ignored. The 0.75 non-synonymous share and 3× per codon are the plan's approximations, not measured. Uniform changes and the chain differ at the same identity by design; the chain's conservation is the calibrated one. Minimizers here rank k-mers at random, where kfp ranks by rarity.

- **2026-11-05: phase 9, combination k-mers.** New panel: pairs or triples of k-mers, hashed as one feature with the gaps left out, sampled at 1 in m, within a distance window with its lower and upper bounds as controls. Distances are between k-mer starts, as kfp's `--min-gap` and `--gap`: consecutive k-mers at least the lower bound apart, first to last at most the upper bound, so a lower bound under k allows overlap. Plots compare single k-mers at the same k and m, at equal postings and with as many letters; two heatmaps show recall and chance across upper bound and sparsity. `measure/combo_grid.py` simulates recall and chance for 24 configurations (`web/data/sim-combo.json`; 38,270 groups, 6,750 read pairs), and points appear at those settings. A first version hashed exact gaps; dropped at the user's request (2026-11-05).
  - **Method.** `model.combo_counts` draws reads from the same conserved blocks and read rate as the k-mer model and counts matching features. Given a read's count C, none survive sampling with chance (1 − 1/m)^C, so one draw serves every m. Work per slider move is capped, so very long reads with many feature sets, and the heatmaps, use fewer reads and are noisier. Chance: with gaps not hashed, a read's feature can meet an index feature from any distance, so it grows with the square of the features per start G, while postings grow with G.
  - **Checks (`checks/test_combo.py`), nothing fitted.** Against simulation, m 1 to 128: recall RMSE 0.041 for singles and pairs, 0.045 for triples, bias −0.009 to −0.028. A single k-mer through the counts matches the chain, RMSE 0.007. Feature counts reproduce kfp's 11, 9 and 36. Simulated reads against kfp's mode table (two shared features on a read, 0.6–0.7 and 0.5–0.6): RMSE 0.053, mean −0.042, though kfp's are exact-gap, on four families and on conserved features. Decoys over composition: 1.11–1.73 across 14 configurations with at least 100 matches, across windows, which supports the G² term.
  - **Window and sparsity, simulated.** At the postings of single k-mers at 1 in 8, recall per read pair at identity 0.5 / 0.6 / 0.7 / 0.75 / 0.8, and chance anchors per pair in frame:

    | Feature | Sparsity | Recall | Chance |
    | --- | --- | --- | --- |
    | k 6 | 1 in 8 | 0.46 / 0.70 / 0.90 / 0.93 / 0.98 | 7.5 |
    | 3-mer pairs, window 3–6 | 1 in 32 | 0.49 / 0.71 / 0.85 / 0.93 / 0.98 | 27 |
    | 3-mer pairs, window 3–13 | 1 in 88 | 0.42 / 0.64 / 0.79 / 0.90 / 0.95 | 66 |
    | 3-mer pairs, window 14–26 | 1 in 104 | 0.27 / 0.50 / 0.68 / 0.84 / 0.92 | 50 |
    | 3-mer pairs, window 1–6 | 1 in 48 | 0.54 / 0.78 / 0.89 / 0.94 / 0.98 | 350 |
    | k 5 | 1 in 8 | 0.59 / 0.82 / 0.94 / 0.97 / 0.98 | 115 |
    | k 8 | 1 in 8 | 0.23 / 0.48 / 0.78 / 0.83 / 0.95 | 0.034 |
    | 4-mer pairs, window 4–6 | 1 in 24 | 0.27 / 0.52 / 0.74 / 0.84 / 0.93 | 0.089 |
    | 4-mer pairs, window 4–12 | 1 in 72 | 0.23 / 0.46 / 0.72 / 0.80 / 0.91 | 0.24 |
    | 4-mer pairs, window 8–16 | 1 in 72 | 0.17 / 0.41 / 0.64 / 0.77 / 0.88 | 0.21 |

    Sparsities not at a power of two are interpolated. **At equal postings a narrow window is better on both counts, and far windows are worst:** widening raises postings by G and chance by G², so m must rise with G while chance per posting still grows. Near windows also hold more conserved letters together. Overlap (lower bound under k) buys recall at much higher chance. No pair window beats single k-mers at equal postings and chance here: 3-mer pairs at 3–6 match k 6's recall at 3.6× its chance, and 4-mer pairs at 4–6 match k 8's at 2.6×.
  - **Limits.** Any-anchor recall only: scores, floors, the multi-hit rule and precision are not modelled for combinations, and those are where a pair's specificity might pay. Simulated reads are real relatives, so hold their indels, but the model's draws have none, which may be part of its low bias. Protein simulation only. Decoy counts are too few to read for 5-mer pairs and above. The panel's postings count feature sets per start; simulated distinct features were 3–9% below that.

- **2026-11-05: combination k-mers simulated on demand.** The combination panel's simulated points now run in the browser at any k, window, read length and alphabet, on the bundled sets, cached like precision results (`live.combo_simulate`). The 24 precomputed configurations moved to `checks/data/sim-combo.json`, as a check only. Recall needs only each query's true group. Chance samples the index's features by hash level, on a share of its groups, and scales up. Very wide windows use fewer read pairs (at least enough to cover each identity bin), and the panel shows the counts. Nucleotide mode has no simulation.
  - **Against the native simulation,** on other queries (`checks/test_live.py`, 18 windows of up to 40 feature sets): recall RMSE 0.016, and chance within 0.99–1.10× where both have at least 50 matches. Chance still rests on few matches at k 5 pairs and above: 9 and 31 matches ran 1.5–1.7× high, and k 6 pairs and k 4 triples had none.
  - **Times in the browser pane, one run each:** 1.4 s for 4-mer pairs at window 4–6, 3.0 s for 3-mer pairs at 3–13 (34% of the index), 20.4 s for 3-mer triples at 3–13 (12%). The slowest setting tried natively, 1,770 feature sets, took 29 s there on 180 read pairs and 0.2% of the index, with too few matches to read chance.
  - **Precision (`live.combo_precision`).** Hits on every group, scored as kfp's keys: composition over the letters a feature fixes, less log2 of the groups holding it, with the charge per further hit counting every feature a read tries. One pass covers 6 sparsity levels upward. Where the index is too large, a share of its groups is held, true groups always among them, and other groups' hits are scaled up. The panel draws precision and recall per read pair by floor beside single k-mers with as many letters at about equal postings, at the precision panel's rule, truth level and identity mix.
    - **Checks.** With one k-mer per feature it reproduces the precision panel's method independently: recall within 0.011–0.014 and precision within a log gap of 0.063–0.291 at settings checked, on 4 hash draws against 8, widest at the highest floors. A share of the index against all of it is checked in `checks/test_live.py`.
    - **Under a score, near pairs about match single k-mers.** Uniform identity mix, correct at x ≥ 0.7, every group over the floor, recall / precision at 30, 40, 60 and 80 bits (postings within about 0.75–1.4× at power-of-two sparsity):

      | Feature | 30 | 40 | 60 | 80 |
      | --- | --- | --- | --- | --- |
      | k 6, 1 in 8 | 0.77 / 0.50 | 0.70 / 0.68 | 0.58 / 0.88 | 0.48 / 0.96 |
      | 3-mer pairs, 3–6, 1 in 32 | 0.74 / 0.57 | 0.66 / 0.76 | 0.51 / 0.92 | 0.38 / 0.96 |
      | 3-mer pairs, 3–13, 1 in 64 | 0.74 / 0.60 | 0.66 / 0.79 | 0.51 / 0.93 | 0.38 / 0.97 |
      | 3-mer pairs, 14–26, 1 in 128 | 0.50 / 0.83 | 0.35 / 0.94 | 0.15 / 0.97 | 0.05 / 0.98 |
      | k 8, 1 in 8 | 0.77 / 0.49 | 0.70 / 0.57 | 0.62 / 0.77 | 0.55 / 0.87 |
      | 4-mer pairs, 4–12, 1 in 64 | 0.75 / 0.53 | 0.68 / 0.66 | 0.59 / 0.84 | 0.51 / 0.93 |
      | k 9, 1 in 4 | 0.84 / 0.34 | 0.77 / 0.45 | 0.74 / 0.57 | 0.70 / 0.66 |
      | 3-mer triples, 3–13, 1 in 128 | 0.83 / 0.38 | 0.75 / 0.52 | 0.68 / 0.69 | 0.64 / 0.80 |

      Reading across at equal recall, near windows gain up to about 0.04 precision on single k-mers at middling recall, about what the wider postings of some rows would buy, and far windows lose. Pair scores also take fewer steps across floors. This is one mini-slice, 4 hash draws and representatives, with reads in frame and no errors; the release has far more relatives per read.

## Licence

GPL-3.0-or-later.
