"""Native simulation grid: real relatives, simulated read pairs and reversed decoys (docs/plan.md).

Each MGnify90 representative is a group. Queries are held-out representatives (their unit leaves
the index) and non-representative members (their unit stays), binned by identity to their nearest
indexed group (overlap alignment, cover >= 0.8), which is the read's true group. Other aligned
groups give the homologue spectrum. Sparsity is hash-sampled, so one pass per k covers every m.

Run from kfp's checkout, for its ``_core`` and on one job with capped threads:
RAYON_NUM_THREADS=2 POLARS_MAX_THREADS=2 uv run --no-sync python <this file> data/mini-slice/members.parquet
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from kmer_functional_profiler import _core

sys.path.insert(0, str(Path(__file__).parent.parent / "web"))
import analyse  # noqa: E402

BINS = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0001])  # lower edges, then the top
RELATIVE_BINS = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
UNALIGNED = len(RELATIVE_BINS) - 1  # a group hit but not among the query's aligned candidates
LEVELS = 8  # m = 1, 2, ... 128
MATE, GAP = 50, 40
MAX_RSS_GB = 6


def code(seqs):
    """Encoded sequences joined by separators."""
    return analyse.ENCODE[np.frombuffer(b"*".join(seqs), np.uint8)]


def level(x):
    """Sampling level of each k-mer: kept at 1 in 2^j when level >= j (splitmix64 of the k-mer)."""
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.minimum(np.log2((z & (~z + np.uint64(1))).astype(float) + (z == 0)).astype(np.int8), LEVELS - 1)


def postings_index(seqs, k):
    """Sorted distinct (k-mer, group), with each k-mer's first row and row count."""
    a = code(seqs)
    x, at = analyse.kmers(a, k)
    group = np.cumsum(a == analyse.SEPARATOR)[at]
    o = np.lexsort((group, x))
    x, group = x[o], group[o]
    keep = np.r_[True, (x[1:] != x[:-1]) | (group[1:] != group[:-1])]
    x, group = x[keep], group[keep]
    keys, first, counts = np.unique(x, return_index=True, return_counts=True)
    return keys, first, counts, group.astype(np.int32)


def lookup(x, owner, index):
    """(owner, group, level) for each of x's postings, keeping each (owner, group)'s highest level."""
    keys, first, counts, group = index
    at = np.searchsorted(keys, x)
    at[at == len(keys)] = 0
    found = keys[at] == x
    at, owner, levels = at[found], owner[found], level(x[found])
    n = counts[at]
    rows = np.repeat(first[at], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    pair_key = np.repeat(owner, n).astype(np.int64) * (int(group.max()) + 2) + group[rows]
    lv = np.repeat(levels, n)
    if not len(pair_key):
        return pair_key, lv
    o = np.argsort(pair_key, kind="stable")
    pair_key, lv = pair_key[o], lv[o]
    starts = np.flatnonzero(np.r_[True, pair_key[1:] != pair_key[:-1]])
    return pair_key[starts], np.maximum.reduceat(lv, starts)


def candidates(queries, reps, top):
    """Each query's ``top`` representatives by shared sampled 5-mers."""
    a = code(reps)
    x, at = analyse.kmers(a, 5)
    group = np.cumsum(a == analyse.SEPARATOR)[at]
    keep = x % 4 == 0
    pairs = np.unique(np.stack([x[keep], group[keep]], 1), axis=0)
    keys, first, counts = np.unique(pairs[:, 0], return_index=True, return_counts=True)
    common = counts > 500
    counts = np.where(common, 0, counts)
    q = code(queries)
    qx, qat = analyse.kmers(q, 5)
    owner = np.cumsum(q == analyse.SEPARATOR)[qat]
    keep = qx % 4 == 0
    both = np.unique(np.stack([qx[keep], owner[keep]], 1), axis=0)
    at = np.searchsorted(keys, both[:, 0])
    at[at == len(keys)] = 0
    found = keys[at] == both[:, 0]
    at, own = at[found], both[found, 1]
    n = counts[at]
    rows = np.repeat(first[at], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    joint = np.repeat(own, n) * (len(reps) + 1) + pairs[rows, 1]
    uniq, shared = np.unique(joint, return_counts=True)
    qs, gs = uniq // (len(reps) + 1), uniq % (len(reps) + 1)
    o = np.lexsort((-shared, qs))
    qs, gs = qs[o], gs[o]
    rank = np.arange(len(qs)) - np.searchsorted(qs, qs)
    return qs[rank < top], gs[rank < top]


def choose(members, a, rng):
    """Queries, indexed representatives, each query's aligned relatives and its true group."""
    reps = members.filter(pl.col("protein_id") == pl.col("cluster_rep"))
    lengths = pl.col("sequence").str.len_chars()
    pool = reps.filter(lengths.is_between(MATE * 2 + GAP, 1000))
    held = pool.sample(min(a.held, pool.height), seed=a.seed)
    kept_units = reps.join(held.select("cluster_rep"), on="cluster_rep", how="anti")
    others = members.filter(pl.col("protein_id") != pl.col("cluster_rep"), lengths.is_between(MATE * 2 + GAP, 1000))
    others = others.join(kept_units.select("cluster_rep"), on="cluster_rep", how="semi")
    picked = others.sample(min(a.member_queries, others.height), seed=a.seed)
    query_rows = pl.concat([held, picked])
    query = [s.encode() for s in query_rows["sequence"]]
    index = [s.encode() for s in kept_units["sequence"]]
    qs, gs = candidates(query, index, a.top)
    short = np.array([len(index[g]) <= 3000 for g in gs])
    qs, gs = qs[short], gs[short]
    t0 = time.perf_counter()
    got = _core.align_identity([query[q] for q in qs], [index[g] for g in gs])
    print(f"aligned {len(qs):,} pairs in {time.perf_counter() - t0:.0f} s", flush=True)
    identity = np.asarray(got["identity"], float)
    cover = np.asarray(got["columns"], float) / np.array([len(query[q]) for q in qs])
    return query, index, qs, gs, identity, cover


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("members")
    p.add_argument("--out", default=str(Path(__file__).parent.parent / "web/data/sim-grid.json"))
    p.add_argument("--held", type=int, default=6000, help="units held out, their representatives queries")
    p.add_argument("--member-queries", type=int, default=1500, help="members of kept units as queries")
    p.add_argument("--per-bin", type=int, default=250)
    p.add_argument("--pairs", type=int, default=10, help="read pairs per query")
    p.add_argument("--top", type=int, default=50, help="relatives aligned per query")
    p.add_argument("--ks", default="5,6,7,8,9,11")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    members = pl.read_parquet(a.members, columns=["protein_id", "cluster_rep", "sequence"])
    query, index, qs, gs, identity, cover = choose(members, a, rng)
    del members

    # True group: the best-identity relative with cover >= 0.8.
    n_q, n_g = len(query), len(index)
    best = np.full(n_q, -1)
    best_id = np.zeros(n_q)
    for i in np.flatnonzero(cover >= 0.8):
        if identity[i] > best_id[qs[i]]:
            best_id[qs[i]], best[qs[i]] = identity[i], gs[i]
    has = best >= 0
    bin_of = np.digitize(best_id, BINS) - 1
    chosen = []
    for b in range(len(BINS) - 1):
        got = np.flatnonzero(has & (bin_of == b))
        chosen.extend(rng.permutation(got)[: a.per_bin].tolist())
    chosen = np.array(sorted(chosen))
    print("queries", n_q, "no relative at cover 0.8:", int((~has).sum()), "per bin",
          np.bincount(bin_of[chosen], minlength=len(BINS) - 1).tolist(), flush=True)

    # Relatives: sorted (query, group) keys with their identity bin (cover >= 0.3).
    ok = cover >= 0.3
    rel_key = qs[ok] * (n_g + 2) + gs[ok]
    rel_bin = np.digitize(identity[ok], RELATIVE_BINS) - 1
    o = np.argsort(rel_key)
    rel_key, rel_bin = rel_key[o], rel_bin[o]
    spectrum = np.zeros((n_q, UNALIGNED + 1))
    np.add.at(spectrum, (qs[ok], np.digitize(identity[ok], RELATIVE_BINS) - 1), 1)
    true_bin = np.digitize(best_id[has], RELATIVE_BINS) - 1
    np.add.at(spectrum, (np.flatnonzero(has), true_bin), -1)  # other groups only

    # Read pairs.
    pair_q, mates, decoys = [], [], []
    for q in chosen:
        s = query[q]
        for _ in range(a.pairs):
            gap = int(rng.integers(0, GAP + 1))
            start = int(rng.integers(0, len(s) - 2 * MATE - gap + 1))
            m1, m2 = s[start : start + MATE], s[start + MATE + gap : start + 2 * MATE + gap]
            pair_q.append(q)
            mates += [m1, m2]
            decoys += [m1[::-1], m2[::-1]]
    pair_q = np.array(pair_q)
    n_pairs = len(pair_q)
    n_bins, n_rel = len(BINS) - 1, UNALIGNED + 1

    rows = []
    for k in map(int, a.ks.split(",")):
        t0 = time.perf_counter()
        idx = postings_index(index, k)
        postings = np.bincount(level(idx[0])[np.repeat(np.arange(len(idx[0])), idx[2])], minlength=LEVELS)
        postings = postings[::-1].cumsum()[::-1]
        hits_true = np.zeros((LEVELS, n_bins))
        other = np.zeros((LEVELS, n_bins, n_rel))
        given_true = np.zeros((LEVELS, n_rel))
        given_miss = np.zeros((LEVELS, n_rel))
        pairs_at_rel_true = np.zeros((LEVELS, n_rel))
        pairs_at_rel_miss = np.zeros((LEVELS, n_rel))
        decoy_anchors = np.zeros(LEVELS)
        scale = n_g + 2
        batch = 2000 if k >= 6 else 500
        pair_ids = np.arange(n_pairs) if k >= 6 else rng.permutation(n_pairs)[: n_pairs // 4]
        for start in range(0, len(pair_ids), batch):
            ids = pair_ids[start : start + batch]
            for kind, source in (("reads", mates), ("decoys", decoys)):
                seqs = [source[2 * i + j] for i in ids for j in (0, 1)]
                encoded = code(seqs)
                x, at = analyse.kmers(encoded, k)
                owner = (np.cumsum(encoded == analyse.SEPARATOR)[at] // 2).astype(np.int64)
                key, lv = lookup(x, owner, idx)
                if not len(key):
                    continue
                pair_of = key // (int(idx[3].max()) + 2)
                group = key % (int(idx[3].max()) + 2)
                if kind == "decoys":
                    for j in range(LEVELS):
                        decoy_anchors[j] += (lv >= j).sum()
                    continue
                query_of = pair_q[ids][pair_of]
                is_true = group == best[query_of]
                bin_q = bin_of[query_of]
                wanted = query_of * scale + group
                rel = np.searchsorted(rel_key, wanted)
                rel[rel == len(rel_key)] = 0
                known = rel_key[rel] == wanted
                x_bin = np.where(known, rel_bin[rel], UNALIGNED)
                best_level = np.full(len(ids), -1, np.int64)
                np.maximum.at(best_level, pair_of[is_true], lv[is_true])
                for j in range(LEVELS):
                    on = lv >= j
                    np.add.at(hits_true[j], bin_q[on & is_true], 1)
                    off = on & ~is_true
                    np.add.at(other[j], (bin_q[off], x_bin[off]), 1)
                    anchored = best_level >= j
                    np.add.at(given_true[j], x_bin[off & anchored[pair_of]], 1)
                    np.add.at(given_miss[j], x_bin[off & ~anchored[pair_of]], 1)
                    spec = spectrum[pair_q[ids]]
                    pairs_at_rel_true[j] += spec[anchored].sum(0)
                    pairs_at_rel_miss[j] += spec[~anchored].sum(0)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
            if rss > MAX_RSS_GB:
                raise SystemExit(f"over {MAX_RSS_GB} GB")
        counted = np.bincount(bin_of[pair_q[pair_ids]], minlength=n_bins)
        for j in range(LEVELS):
            rows.append({
                "k": k,
                "m": 2**j,
                "postings": int(postings[j]),
                "pairs": counted.tolist(),
                "recall": (hits_true[j] / np.maximum(counted, 1)).tolist(),
                "other_per_pair": (other[j] / np.maximum(counted, 1)[:, None]).tolist(),
                "other_given_anchor": given_true[j].tolist(),
                "relatives_given_anchor": pairs_at_rel_true[j].tolist(),
                "other_given_miss": given_miss[j].tolist(),
                "relatives_given_miss": pairs_at_rel_miss[j].tolist(),
                "decoy_anchors_per_pair": float(decoy_anchors[j] / len(pair_ids)),
            })
        print(f"k {k}: {time.perf_counter() - t0:.0f} s, peak {rss:.1f} GB, "
              f"recall at 0.9-0.95, m 8: {rows[-5]['recall'][6]:.3f}", flush=True)
    chosen_bins = bin_of[chosen]
    per_query = [spectrum[chosen[chosen_bins == b]].mean(0).tolist() if (chosen_bins == b).any() else [0] * n_rel for b in range(n_bins)]
    Path(a.out).write_text(json.dumps({
        "source": str(a.members),
        "groups": n_g,
        "queries": len(chosen),
        "pool": n_q,
        "no_relative": int((~has).sum()),
        "pairs": n_pairs,
        "mate": MATE,
        "gap": GAP,
        "top": a.top,
        "seed": a.seed,
        "identity_bins": BINS.tolist(),
        "mean_identity": [float(best_id[chosen[chosen_bins == b]].mean()) if (chosen_bins == b).any() else None for b in range(n_bins)],
        "relative_bins": RELATIVE_BINS.tolist(),
        "relatives_per_query": per_query,
        "rows": rows,
    }))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
