"""Can precision come from shared k-mer counts? (docs/plan.md, Progress log.)

M = distinct k-mers a read pair shares with a group at full density. Sampling keeps each at 1/m, so
hits ~ Binomial(M, 1/m) for any m. Compares that with real hash sampling on the same reads, for
counts, hit thresholds, truth levels and best-group assignment, and tests how few reads are enough.

Run from kfp's checkout, one job with capped threads:
RAYON_NUM_THREADS=2 POLARS_MAX_THREADS=2 uv run --no-sync python <this file> members.parquet
"""

import argparse
import resource
import sys
from math import comb
from pathlib import Path

import numpy as np
import polars as pl
from kmer_functional_profiler import _core

sys.path.insert(0, str(Path(__file__).parent))
import sim_grid as sg  # noqa: E402

KS = (7, 9, 11)
LEVELS = (3, 5, 7)  # m = 8, 32, 128
HITS = (1, 2, 3, 5)
SALTS = 8


def rows_for(mates, k, index, salt=0):
    """(pair, group, per-level distinct hit counts [rows x 8]) of each read pair on each group."""
    encoded = sg.code(mates)
    x, at = sg.analyse.kmers(encoded, k)
    owner = (np.cumsum(encoded == sg.analyse.SEPARATOR)[at] // 2).astype(np.int64)
    o = np.lexsort((x, owner))
    x, owner = x[o], owner[o]
    distinct = np.r_[True, (x[1:] != x[:-1]) | (owner[1:] != owner[:-1])]
    x, owner = x[distinct], owner[distinct]
    keys, first, counts, group = index
    pos = np.searchsorted(keys, x)
    pos[pos == len(keys)] = 0
    found = keys[pos] == x
    pos, owner, lv = pos[found], owner[found], sg.level(x[found] ^ salt)
    n = counts[pos]
    rows = np.repeat(first[pos], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    scale = int(group.max()) + 2
    uniq, inverse = np.unique(np.repeat(owner, n) * scale + group[rows], return_inverse=True)
    table = np.zeros((len(uniq), 8), np.int32)
    np.add.at(table, (inverse, np.repeat(lv, n)), 1)
    return uniq // scale, uniq % scale, table[:, ::-1].cumsum(1)[:, ::-1]


def at_least(counts, m, h):
    """P(Binomial(counts, 1/m) >= h), for an integer array of counts."""
    top = int(counts.max()) if len(counts) else 0
    table = np.zeros(top + 1)
    p = 1 / m
    for c in range(top + 1):
        table[c] = 1 - sum(comb(c, i) * p**i * (1 - p) ** (c - i) for i in range(min(h, c + 1)))
    return table[counts]


def best_group(pair, hits, n_pairs):
    """Rows holding their pair's largest hit count (ties kept), among pairs with any hit."""
    top = np.zeros(n_pairs, np.int32)
    np.maximum.at(top, pair, hits)
    return (hits == top[pair]) & (hits > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("members")
    p.add_argument("--held", type=int, default=6000)
    p.add_argument("--member-queries", type=int, default=1500)
    p.add_argument("--per-bin", type=int, default=150)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--unit-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    members = pl.read_parquet(a.members, columns=["protein_id", "cluster_rep", "sequence"])
    query, index, qs, gs, identity, cover = sg.choose(members, a, rng)
    del members
    n_q, n_g = len(query), len(index)
    best, best_id = np.full(n_q, -1), np.zeros(n_q)
    for i in np.flatnonzero(cover >= 0.8):
        if identity[i] > best_id[qs[i]]:
            best_id[qs[i]], best[qs[i]] = identity[i], gs[i]
    bin_of = np.digitize(best_id, sg.BINS) - 1
    chosen = []
    for b in range(len(sg.BINS) - 1):
        got = np.flatnonzero((best >= 0) & (bin_of == b))
        chosen.extend(rng.permutation(got)[: a.per_bin].tolist())
    pair_q, mates = [], []
    for q in sorted(chosen):
        s = query[q]
        for _ in range(a.pairs):
            gap = int(rng.integers(0, sg.GAP + 1))
            start = int(rng.integers(0, len(s) - 2 * sg.MATE - gap + 1))
            pair_q.append(q)
            mates += [s[start : start + sg.MATE], s[start + sg.MATE + gap : start + 2 * sg.MATE + gap]]
    pair_q = np.array(pair_q)
    n_pairs = len(pair_q)
    print(f"{n_pairs} pairs, {n_g} groups", flush=True)

    aligned = qs.astype(np.int64) * (n_g + 2) + gs
    o = np.argsort(aligned)
    aligned, aligned_x = aligned[o], identity[o]

    for k in KS:
        pair, group, levels = rows_for(mates, k, sg.postings_index(index, k, sg.analyse.ENCODE))
        M = levels[:, 0]
        q_of = pair_q[pair]
        true = group == best[q_of]
        wanted = q_of.astype(np.int64) * (n_g + 2) + group
        at = np.searchsorted(aligned, wanted)
        at[at == len(aligned)] = 0
        known = aligned[at] == wanted
        x = np.where(known, aligned_x[at], np.nan)
        if k == 9:  # whole-protein identity for every row, for truth levels
            distinct, inverse = np.unique(wanted[~known], return_inverse=True)
            got = _core.align_identity(
                [query[int(v // (n_g + 2))] for v in distinct], [index[int(v % (n_g + 2))] for v in distinct]
            )
            x[~known] = np.asarray(got["identity"], float)[inverse]
            print(f"aligned {len(distinct):,} more", flush=True)
        print(f"\nk{k}: {len(pair):,} rows, {len(pair) / n_pairs:.0f} groups sharing >= 1 k-mer per pair, "
              f"mean M true {M[true].mean():.1f}, other {M[~true].mean():.2f}", flush=True)
        print(" m, h: recall real / binomial; others per pair real / binomial; precision real / binomial"
              + ("; precision x>=.7 real / binomial" if k == 9 else ""))
        for level in LEVELS:
            m = 2**level
            hits = levels[:, level]
            for h in HITS:
                prob = at_least(M, m, h)
                real = hits >= h
                # Recall is P(a pair anchors the true group), one row per pair at most.
                r_tp, b_tp = real[true].sum() / n_pairs, prob[true].sum() / n_pairs
                r_fp, b_fp = real[~true].sum() / n_pairs, prob[~true].sum() / n_pairs
                line = (f"  {m:4d} {h}: {r_tp:.3f} / {b_tp:.3f}; {r_fp:7.2f} / {b_fp:7.2f}; "
                        f"{r_tp / max(r_tp + r_fp, 1e-9):.3f} / {b_tp / max(b_tp + b_fp, 1e-9):.3f}")
                if k == 9:
                    ok = true | (x >= 0.7)
                    line += (f"; {real[ok].sum() / max(real.sum(), 1):.3f} / "
                             f"{prob[ok].sum() / max(prob.sum(), 1e-9):.3f}")
                print(line, flush=True)
        # Best group per pair. Real sampling shares decisions across groups; thinning here does not.
        print(" best group per pair, m: assigned, precision, true group in the best set: real / independent thinning")
        for level in LEVELS:
            m = 2**level
            real = best_group(pair, levels[:, level], n_pairs)
            draws = []
            for seed in range(10):
                thin = np.random.default_rng(seed).binomial(M, 1 / m)
                keep = best_group(pair, thin, n_pairs)
                draws.append(((keep & true).sum() / max(keep.sum(), 1), (keep & true).sum() / n_pairs, keep.sum() / n_pairs))
            d = np.mean(draws, 0)
            print(f"  {m:4d}: groups per pair {real.sum() / n_pairs:.2f} / {d[2]:.2f}; "
                  f"precision {(real & true).sum() / max(real.sum(), 1):.3f} / {d[0]:.3f}; "
                  f"recall {(real & true).sum() / n_pairs:.3f} / {d[1]:.3f}", flush=True)
        # One hash draw, or the expectation? Real sampling over salts against the binomial.
        idx = sg.postings_index(index, k, sg.analyse.ENCODE)
        draws = {}
        for salt in range(1, SALTS + 1):
            pa, gr, lv = rows_for(mates, k, idx, salt=salt * 0x51ED27)
            tr = gr == best[pair_q[pa]]
            for level in LEVELS[:2]:
                for h in (1, 3):
                    on = lv[:, level] >= h
                    tp, fp = (on & tr).sum() / n_pairs, (on & ~tr).sum() / n_pairs
                    draws.setdefault((level, h), []).append((tp, fp, tp / max(tp + fp, 1e-9)))
                bg = best_group(pa, lv[:, level], n_pairs)
                draws.setdefault((level, "best"), []).append(
                    ((bg & tr).sum() / n_pairs, (bg & ~tr).sum() / n_pairs, (bg & tr).sum() / max(bg.sum(), 1))
                )
        print(f" over {SALTS} hash salts: recall, others, precision as mean (sd); binomial alongside")
        for (level, h), v in draws.items():
            mean, sd = np.mean(v, 0), np.std(v, 0)
            m = 2**level
            if h == "best":
                print(f"  m {m:3d} best group: {mean[0]:.3f} ({sd[0]:.3f}), {mean[1]:.2f} ({sd[1]:.2f}), {mean[2]:.3f} ({sd[2]:.3f})", flush=True)
                continue
            prob = at_least(M, m, h)
            b_tp, b_fp = prob[true].sum() / n_pairs, prob[~true].sum() / n_pairs
            print(f"  m {m:3d} h {h}: {mean[0]:.3f} ({sd[0]:.3f}), {mean[1]:.2f} ({sd[1]:.2f}), {mean[2]:.3f} ({sd[2]:.3f}); "
                  f"binomial {b_tp:.3f}, {b_fp:.2f}, {b_tp / max(b_tp + b_fp, 1e-9):.3f}", flush=True)
        # How few reads are enough? Precision at m 8, h 2, from query-level subsamples.
        queries = np.unique(q_of)
        prob = at_least(M, 8, 2)
        full = prob[true].sum() / prob.sum()
        by_q = np.searchsorted(queries, q_of)
        tp = np.bincount(by_q, prob * true, len(queries))
        total = np.bincount(by_q, prob, len(queries))
        print(f" precision at m 8, h 2 is {full:.4f}; relative spread from subsampled queries:")
        for n in (25, 50, 100, 200, 400):
            est = []
            for seed in range(200):
                s = np.random.default_rng(seed).choice(len(queries), n, replace=True)
                est.append(tp[s].sum() / max(total[s].sum(), 1e-9))
            print(f"  {n:4d} queries: sd / mean {np.std(est) / np.mean(est):.2f}")
        print(f" M table size: {len(np.unique(np.stack([bin_of[q_of], np.minimum(M, 60), true], 1), axis=0))} cells (identity bin, M, truth)", flush=True)
    print("peak GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, flush=True)


if __name__ == "__main__":
    main()
