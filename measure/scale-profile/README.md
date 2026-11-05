# Scale profile

Measures how statistics grow with a protein database, for `docs/plan.md`, Scaling to full size: a
small set cannot be extrapolated to MGnify, so the page interpolates between these.

- **Rungs:** uniform samples of 10⁵, 10⁶, 10⁷ and 10⁸ proteins, chosen by a hash of `protein_id`, so
  each holds the one below. Phase 0's `cluster_rep % sample` keeps whole clusters and cannot measure
  how units grow.
- **Anchors:** proteins, residues, MGnify90 units and singletons over the whole release.
- **Per rung:** units, singletons, lengths, composition; per k, distinct k-mers, Good–Turing counts,
  holders, the hot tail, decoys and matches between random halves, from a hash slice of the k-mers
  sized to memory; and the homologue spectrum, 2,000 held-out proteins aligned with DIAMOND.
- **`check.tsv`:** each statistic's largest rung predicted from the others. The error is the band for
  the page's curves.

On HPC: see the header of `run.sh`. Test on a local members table:

```bash
nextflow run measure/scale-profile/main.nf -profile test,docker --store "$PWD/../kmer-functional-profiler/data/mini-slice/members.parquet" --venv "$PWD/../kmer-functional-profiler/.venv" -work-dir measure/scale-profile/work
```

Not included: decoy tails per (k, m) on a real index, sticky leaves and key retention, which need
indexes built on the rungs.
