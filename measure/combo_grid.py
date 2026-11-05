"""Simulation for combination k-mers (docs/plan.md, phase 9): n k-mers in order, consecutive starts at
least ``lo`` apart and first to last at most ``hi``, hashed as one feature without the gaps, so a pair
matches at any distance in the window, then sampled at 1 in m. Same relatives and read pairs as
sim_grid.py. Recall needs only the true group, so every sparsity comes from one pass; chance is read
from the index's features at one sparse level and scaled, since an index of them all would be too large.

Run from kfp's checkout, one job with capped threads:
RAYON_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python <this file> data/mini-slice/members.parquet --out web/data/sim-combo.json
"""

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import sim_grid as sg  # noqa: E402

sg.LEVELS = 14
LEVELS = 8  # m = 1 to 128
NO_RELATIVE = len(sg.BINS) - 1
INDEX_FEATURES = 15_000_000  # features kept from the index for the chance estimate
MAX_RSS_GB = 10
CONFIGS = [(5, 1, 0, 0), (6, 1, 0, 0), (7, 1, 0, 0), (8, 1, 0, 0), (9, 1, 0, 0),  # (k, n, lo, hi)
           (3, 2, 3, 13), (3, 2, 1, 6), (3, 2, 1, 13), (3, 2, 3, 6), (3, 2, 7, 13), (3, 2, 14, 26), (3, 2, 3, 26),
           (4, 2, 4, 12), (4, 2, 1, 12), (4, 2, 8, 16), (4, 2, 4, 26), (4, 2, 4, 6),
           (5, 2, 5, 11), (5, 2, 5, 30), (6, 2, 6, 10),
           (3, 3, 3, 13), (3, 3, 1, 8), (4, 3, 4, 12), (4, 3, 4, 20)]  # fmt: skip


def watchdog():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30 > MAX_RSS_GB:
            print(f"over {MAX_RSS_GB} GB, stopping", flush=True)
            os._exit(3)
        time.sleep(1)


class Kmers:
    """Each position's k-mer code (-1 if none) and sequence, over sequences joined by separators."""

    def __init__(self, seqs, k):
        a = sg.code(seqs)
        x, at = sg.analyse.kmers(a, k)
        self.code = np.full(len(a), -1, np.int64)
        self.code[at] = x
        self.owner = np.cumsum(a == sg.analyse.SEPARATOR).astype(np.int32)

    def features(self, offsets):
        """(owner, feature hash) for one set of offsets."""
        last = offsets[-1] if offsets else 0
        i = np.arange(len(self.code) - last)
        ok = self.code[i] >= 0
        h = self.code[i].astype(np.uint64)
        for o in offsets:
            ok &= (self.code[i + o] >= 0) & (self.owner[i + o] == self.owner[i])
            h = h * np.uint64(0x100000001B3) + self.code[i + o].astype(np.uint64)
        return self.owner[i[ok]], h[ok]


def all_features(kmers, sets, keep_level=0):
    owners, hashes = [], []
    for offsets in sets:
        o, h = kmers.features(offsets)
        if keep_level:
            kept = sg.level(h) >= keep_level
            o, h = o[kept], h[kept]
        owners.append(o)
        hashes.append(h)
    return pl.DataFrame({"owner": np.concatenate(owners), "h": np.concatenate(hashes)}).unique()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("members")
    p.add_argument("--out", required=True)
    p.add_argument("--held", type=int, default=6000)
    p.add_argument("--member-queries", type=int, default=1500)
    p.add_argument("--per-bin", type=int, default=150)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--unit-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=4)
    a = p.parse_args()
    threading.Thread(target=watchdog, daemon=True).start()
    rng = np.random.default_rng(a.seed)
    members = pl.read_parquet(a.members, columns=["protein_id", "cluster_rep", "sequence"])
    query, index, qs, gs, identity, cover = sg.choose(members, a, rng)
    del members
    best, best_x = np.full(len(query), -1), np.zeros(len(query))
    for i in np.flatnonzero(cover >= 0.8):
        if identity[i] > best_x[qs[i]]:
            best_x[qs[i]], best[qs[i]] = identity[i], gs[i]
    bin_of = np.digitize(best_x, sg.BINS) - 1
    bin_of[(best < 0) | (bin_of < 0)] = NO_RELATIVE
    chosen = sorted(
        i for b in range(NO_RELATIVE) for i in rng.permutation(np.flatnonzero(bin_of == b))[: a.per_bin].tolist()
    )  # fmt: skip
    pair_q, mates, decoys = [], [], []
    for q in chosen:
        s = query[q]
        for _ in range(a.pairs):
            gap = int(rng.integers(0, sg.GAP + 1))
            start = int(rng.integers(0, len(s) - 2 * sg.MATE - gap + 1))
            m1, m2 = s[start : start + sg.MATE], s[start + sg.MATE + gap : start + 2 * sg.MATE + gap]
            pair_q.append(q)
            mates += [m1, m2]
            decoys += [m1[::-1], m2[::-1]]
    pair_q = np.array(pair_q)
    n_pairs, pair_bin = len(pair_q), bin_of[pair_q]
    groups_wanted = np.unique(best[pair_q])
    group_seqs = [index[g] for g in groups_wanted]
    true_group = np.searchsorted(groups_wanted, best[pair_q])
    residues = sum(map(len, index))
    print(f"{n_pairs} read pairs, {len(index)} groups, {residues:,} residues", flush=True)
    out = {
        "source": a.members, "groups": len(index), "residues": residues, "pairs": n_pairs, "mate": sg.MATE,
        "gap": sg.GAP, "bins": sg.BINS.tolist(), "seed": a.seed, "rows": [],
    }  # fmt: skip
    for k, n, lo, hi in CONFIGS:
        t0 = time.perf_counter()
        sets = sg.model.combo_sets(n, lo, hi)
        reads, group_kmers = Kmers(mates, k), Kmers(group_seqs, k)
        read_f = all_features(reads, sets)
        group_f = all_features(group_kmers, sets)
        joined = read_f.with_columns(
            pair=pl.col("owner") // 2, group=pl.Series(true_group[read_f["owner"].to_numpy() // 2])
        ).join(group_f.rename({"owner": "group"}), on=["group", "h"])
        level = sg.level(joined["h"].to_numpy())
        pair_hits = np.zeros((LEVELS, n_pairs), np.int32)
        mate_hits = np.zeros((LEVELS, 2 * n_pairs), np.int32)
        owner = joined["owner"].to_numpy()
        for j in range(LEVELS):
            on = level >= j
            mate_hits[j] = np.bincount(owner[on], minlength=2 * n_pairs)
            pair_hits[j] = mate_hits[j, ::2] + mate_hits[j, 1::2]
        by_bin = lambda x: (np.bincount(pair_bin, weights=x, minlength=NO_RELATIVE)[:NO_RELATIVE]  # noqa: E731
                            / np.maximum(np.bincount(pair_bin, minlength=NO_RELATIVE)[:NO_RELATIVE], 1))  # fmt: skip
        total = len(sets) * residues
        keep_level = max(int(np.ceil(np.log2(max(total / INDEX_FEATURES, 1)))), 0)
        index_f = all_features(Kmers(index, k), sets, keep_level)
        decoy_f = all_features(Kmers(decoys, k), sets, keep_level)
        matches = decoy_f.join(index_f.rename({"owner": "group"}), on="h").height
        row = {
            "k": k, "n": n, "lo": lo, "hi": hi, "feature_sets": len(sets), "index_features": index_f.height * 2**keep_level,
            "recall": [by_bin(pair_hits[j] > 0).round(4).tolist() for j in range(LEVELS)],
            "two_on_a_read": [by_bin((mate_hits[j, ::2] > 1).astype(float)).round(4).tolist() for j in range(LEVELS)],
            "chance_keep_level": keep_level, "chance_matches": matches, "chance_per_pair_m1": matches * 2**keep_level / n_pairs,
        }  # fmt: skip
        out["rows"].append(row)
        Path(a.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
        print(f"k{k} n{n} window {lo}-{hi}: {len(sets)} sets, recall at 0.8-0.85, m 8 {row['recall'][3][4]:.3f}; "
              f"{matches} decoy matches at 1 in {2**keep_level}; {time.perf_counter() - t0:.0f} s, peak "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
