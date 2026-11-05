"""What the page needs to compute precision on demand (web/live.py), per index: the index's sequences,
the query proteins with their identity bins and true groups, and aligned identities for every
(query, group) sharing a protein 7-mer. Nothing here depends on k, m or read length.

Run from kfp's checkout, one job with capped threads:
RAYON_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python <this file> members.parquet --out web/data/live-x.npz
"""

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from kmer_functional_profiler import _core

sys.path.insert(0, str(Path(__file__).parent))
import sim_grid as sg  # noqa: E402

NO_RELATIVE = len(sg.BINS) - 1
SHARED_K = 7
MAX_ALIGN = 3000  # longer groups are left out of alignment, as in sim_grid.choose


def shared_groups(queries, index):
    """Distinct (query, group) sharing a k-mer."""
    keys, first, counts, group = sg.postings_index(index, SHARED_K, sg.analyse.ENCODE)
    a = sg.code(queries)
    x, at = sg.analyse.kmers(a, SHARED_K)
    owner = np.cumsum(a == sg.analyse.SEPARATOR)[at]
    both = np.unique(np.stack([x, owner], 1), axis=0)
    pos = np.searchsorted(keys, both[:, 0])
    pos[pos == len(keys)] = 0
    found = keys[pos] == both[:, 0]
    pos, owner = pos[found], both[found, 1]
    n = counts[pos]
    rows = np.repeat(first[pos], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    scale = len(index) + 1
    uniq = np.unique(np.repeat(owner, n) * scale + group[rows])
    return uniq // scale, uniq % scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("members")
    p.add_argument("--out", required=True)
    p.add_argument("--held", type=int, default=6000)
    p.add_argument("--member-queries", type=int, default=1500)
    p.add_argument("--per-bin", type=int, default=150)
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--unit-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3, help="not the tables' seeds, so they stay a check")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    members = pl.read_parquet(a.members, columns=["protein_id", "cluster_rep", "sequence"])
    query, index, qs, gs, identity, cover = sg.choose(members, a, rng)
    del members
    best, best_x = np.full(len(query), -1), np.zeros(len(query))
    for i in np.flatnonzero(cover >= 0.8):
        if identity[i] > best_x[qs[i]]:
            best_x[qs[i]], best[qs[i]] = identity[i], gs[i]
    bin_of = np.digitize(best_x, sg.BINS) - 1
    bin_of[best < 0] = NO_RELATIVE
    bin_of[bin_of < 0] = NO_RELATIVE  # a relative below the lowest bin
    best[bin_of == NO_RELATIVE] = -1
    chosen = np.array(sorted(
        i for b in range(NO_RELATIVE + 1)
        for i in rng.permutation(np.flatnonzero(bin_of == b))[: a.per_bin].tolist()
    ))  # fmt: skip
    new_of = np.full(len(query), -1)
    new_of[chosen] = np.arange(len(chosen))
    chosen_seqs = [query[i] for i in chosen]
    keep = new_of[qs] >= 0
    q0, g0, x0 = new_of[qs[keep]], gs[keep], identity[keep]
    t0 = time.perf_counter()
    q1, g1 = shared_groups(chosen_seqs, index)
    scale = len(index) + 1
    new = ~np.isin(q1 * scale + g1, q0 * scale + g0)
    short = np.array([len(index[g]) <= MAX_ALIGN for g in g1])
    q1, g1 = q1[new & short], g1[new & short]
    print(f"aligning {len(q1):,} more pairs", flush=True)
    got = _core.align_identity([chosen_seqs[q] for q in q1], [index[g] for g in g1])
    x1 = np.asarray(got["identity"], float)
    print(f"{time.perf_counter() - t0:.0f} s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB", flush=True)
    assert len(index) < 2**32 and len(chosen) < 2**16
    np.savez_compressed(
        a.out,
        index=np.frombuffer(b"*".join(index), np.uint8),
        queries=np.frombuffer(b"*".join(chosen_seqs), np.uint8),
        bin=bin_of[chosen].astype(np.uint8),
        best=best[chosen].astype(np.int32),
        aligned_q=np.r_[q0, q1].astype(np.uint16),
        aligned_g=np.r_[g0, g1].astype(np.uint32),
        aligned_x=np.round(np.clip(np.r_[x0, x1], 0, 1) * 250).astype(np.uint8),
        groups=np.array(len(index)),
    )
    print(f"{a.out}: {len(chosen)} queries, {len(index)} groups, {len(q0) + len(q1):,} aligned, "
          f"{Path(a.out).stat().st_size / 1e6:.1f} MB", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
