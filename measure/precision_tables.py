"""Shared k-mer tables for the page's precision (docs/plan.md, phase 4).

M = distinct k-mers a read pair shares with a group at full density. Sampling keeps each at 1/m, so
hits ~ Binomial(M, 1/m) for any m. Writes M's counts by (query identity bin, relative identity band or
truth) per k, on reads and reversed decoys, with real hash sampling and its spread alongside to test
them. Queries with no relative at cover 0.8 form their own bin.

Run from kfp's checkout, one job with capped threads:
RAYON_NUM_THREADS=2 POLARS_MAX_THREADS=2 uv run --no-sync python <this file> members.parquet --out x.json
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

sys.path.insert(0, str(Path(__file__).parent))
import sim_grid as sg  # noqa: E402

NO_RELATIVE = len(sg.BINS) - 1  # the identity bin of queries with no relative at cover 0.8
X_EDGES = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
TRUE_BAND, UNKNOWN_BAND = -1, len(X_EDGES) - 1  # a row's relative-identity band, or these
ALIGN_FROM_K = 7  # below this, too many rows to align: their identity stays unknown
LEVELS_MAX = 14  # real sampling is recorded for m = 1 to 8,192
sg.LEVELS = LEVELS_MAX
HITS = (1, 2, 3, 5)
SALTS = 8
FLOORS = (0, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200)  # information bits; 0 is no floor
N_BANDS = len(X_EDGES) + 1  # the true group first, then relative identity bands, then unknown
TRANSLATED_FRAMES = 6  # kfp's charge counts every frame's windows, about 9 bits per read pair
MARGINS = (0, 3, 10)  # best-group rule: bits below a read's top score still kept (kfp's default is 3)
MAX_RSS_GB = 8


def key_bits(index, k, seqs):
    """Each indexed k-mer's information, as kfp's keys: -log2 q0 - log2 g, at least 0, with q0 its
    chance from residue composition and g the groups holding it."""
    keys, _, holders, _ = index
    letters = np.bincount(sg.code(seqs), minlength=20)[:20].astype(float)
    log_f = np.log2(np.maximum(letters / letters.sum(), 1e-12))
    bits = np.zeros(len(keys))
    rest = keys.copy()
    for _ in range(k):
        bits -= log_f[rest % 21]
        rest //= 21
    return np.maximum(bits - np.log2(holders), 0)


def rows_for(mates, k, index, salt=0, bits=None):
    """(pair, group, per-level distinct hit counts [rows x levels]) of each read pair on each group, and
    with ``bits`` the same levels' summed key bits."""
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
    table = np.zeros((len(uniq), LEVELS_MAX), np.int32)
    np.add.at(table, (inverse, np.repeat(lv, n)), 1)
    counts_up = table[:, ::-1].cumsum(1)[:, ::-1]
    if bits is None:
        return uniq // scale, uniq % scale, counts_up
    summed = np.zeros((len(uniq), LEVELS_MAX))
    np.add.at(summed, (inverse, np.repeat(lv, n)), np.repeat(bits[pos], n))
    return uniq // scale, uniq % scale, counts_up, summed[:, ::-1].cumsum(1)[:, ::-1]


def scores(levels, summed, level, charge):
    """kfp's evidence: summed bits less the charge per hit after the first; rows with no hit -inf."""
    hits = levels[:, level]
    return np.where(hits > 0, summed[:, level] - np.maximum(hits - 1, 0) * charge, -np.inf)


def top_of_pair(pair, score):
    """Each row's read pair's top score. Rows are sorted by pair."""
    starts = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
    tops = np.maximum.reduceat(score, starts)
    return np.repeat(tops, np.diff(np.r_[starts, len(pair)]))


def per_target(pair, group, true, on, pair_q, n_queries):
    """(true targets called, false groups called) per query: a group is called by any read pair."""
    q = pair_q[pair[on]]
    key = q.astype(np.int64) * (int(group.max()) + 2) + group[on]
    _, first = np.unique(key, return_index=True)
    return true[on][first].sum() / n_queries, (~true[on][first]).sum() / n_queries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("members")
    p.add_argument("--out", required=True)
    p.add_argument("--held", type=int, default=6000)
    p.add_argument("--member-queries", type=int, default=1500)
    p.add_argument("--per-bin", type=int, default=150, help="queries per identity bin, and with no relative")
    p.add_argument("--pairs", type=int, default=3, help="read pairs per query")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--ks", default="13,11,9,8,7,6,5", help="largest first, so alignments are reused")
    p.add_argument("--unit-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    members = pl.read_parquet(a.members, columns=["protein_id", "cluster_rep", "sequence"])
    query, index, qs, gs, identity, cover = sg.choose(members, a, rng)
    del members
    n_q, n_g = len(query), len(index)
    scale = n_g + 2
    best, best_id = np.full(n_q, -1), np.zeros(n_q)
    for i in np.flatnonzero(cover >= 0.8):
        if identity[i] > best_id[qs[i]]:
            best_id[qs[i]], best[qs[i]] = identity[i], gs[i]
    bin_of = np.digitize(best_id, sg.BINS) - 1
    bin_of[best < 0] = NO_RELATIVE
    chosen = []
    for b in range(NO_RELATIVE + 1):
        got = np.flatnonzero(bin_of == b)
        chosen.extend(rng.permutation(got)[: a.per_bin].tolist())
    chosen = sorted(chosen)
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
    n_pairs = len(pair_q)
    pair_bin = bin_of[pair_q]
    pairs_by_bin = np.bincount(pair_bin, minlength=NO_RELATIVE + 1)
    queries_by_bin = np.bincount(bin_of[chosen], minlength=NO_RELATIVE + 1)
    print(f"{n_pairs} pairs, {n_g} groups, pairs by bin {pairs_by_bin.tolist()}", flush=True)
    relative = pair_bin != NO_RELATIVE

    aligned = qs.astype(np.int64) * scale + gs
    o = np.argsort(aligned)
    aligned, aligned_x = aligned[o], identity[o]
    out = {
        "source": a.members, "groups": n_g, "pairs": n_pairs, "pairs_per_query": a.pairs, "mate": sg.MATE,
        "seed": a.seed, "identity_bins": sg.BINS.tolist(), "x_edges": X_EDGES.tolist(),
        "pairs_by_bin": pairs_by_bin.tolist(), "queries_by_bin": queries_by_bin.tolist(), "k": {},
        "floors": list(FLOORS), "salts": SALTS, "n_bands": N_BANDS, "margins": list(MARGINS),
    }
    for k in map(int, a.ks.split(",")):
        t0 = time.perf_counter()
        idx = sg.postings_index(index, k, sg.analyse.ENCODE)
        pair, group, levels = rows_for(mates, k, idx)
        M = levels[:, 0]
        q_of = pair_q[pair]
        true = group == best[q_of]
        wanted = q_of.astype(np.int64) * scale + group
        at = np.searchsorted(aligned, wanted)
        at[at == len(aligned)] = 0
        known = aligned[at] == wanted
        if k >= ALIGN_FROM_K and (~known).any():  # identity for every row, for truth levels
            distinct = np.unique(wanted[~known])
            got = _core.align_identity(
                [query[int(v // scale)] for v in distinct], [index[int(v % scale)] for v in distinct]
            )
            aligned = np.concatenate([aligned, distinct])
            aligned_x = np.concatenate([aligned_x, np.asarray(got["identity"], float)])
            o = np.argsort(aligned)
            aligned, aligned_x = aligned[o], aligned_x[o]
            at = np.searchsorted(aligned, wanted)
            at[at == len(aligned)] = 0
            known = aligned[at] == wanted
            print(f"k{k}: aligned {len(distinct):,} more", flush=True)
        x = np.where(known, aligned_x[at], np.nan)
        band = np.where(true, TRUE_BAND, np.where(known, np.digitize(x, X_EDGES) - 1, UNKNOWN_BAND))
        cells = np.stack([pair_bin[pair], band, M], 1)
        keys, counts = np.unique(cells, axis=0, return_counts=True)
        table = [[*map(int, c), int(n)] for c, n in zip(keys, counts, strict=True)]

        # Each query's M on its true group per read pair: reads of one target share its conservation.
        true_m = np.zeros(n_pairs, np.int32)
        true_m[pair[true]] = M[true]
        by_query = true_m.reshape(-1, a.pairs)
        per_query = np.column_stack([bin_of[chosen], by_query])[np.array(bin_of[chosen]) != NO_RELATIVE]

        d_pair, _, d_levels = rows_for(decoys, k, idx)
        d_keys, d_counts = np.unique(d_levels[:, 0], return_counts=True)

        real = {}
        for level in range(LEVELS_MAX):
            hits = levels[:, level]
            for h in HITS:
                on = hits >= h
                ok = true | (x >= 0.7)
                t_true, t_false = per_target(pair, group, true, on, pair_q, len(chosen))
                real[f"{level},{h}"] = {
                    "recall": float((on & true & relative[pair]).sum() / max(relative.sum(), 1)),
                    "others": float((on & ~true & relative[pair]).sum() / max(relative.sum(), 1)),
                    "precision": float((on & true).sum() / max(on.sum(), 1)),
                    "precision_70": float((on & ok).sum() / max(on.sum(), 1)) if k >= ALIGN_FROM_K else None,
                    "target_true": float(t_true),
                    "target_false": float(t_false),
                    "decoys": float((d_levels[:, level] >= h).sum() / n_pairs),
                }
        # One index is one hash draw: the spread of precision over salts.
        spread, salted = {}, {}
        bits = key_bits(idx, k, index)
        charge = float(np.log2(2 * TRANSLATED_FRAMES * (sg.MATE - k + 1)))
        band_at = np.where(band == TRUE_BAND, 0, band + 1)
        cell = pair_bin[pair] * N_BANDS + band_at
        n_bins, n_levels, n_floors = NO_RELATIVE + 1, LEVELS_MAX, len(FLOORS)
        cube = np.zeros((n_levels, n_floors, n_bins * N_BANDS), np.int64)
        best_cube = np.zeros((len(MARGINS), n_levels, n_floors, n_bins * N_BANDS), np.int64)
        hit_bits = np.zeros((n_levels, 2))
        query_pass = np.zeros((n_levels, n_floors, n_bins, a.pairs + 1), np.int64)
        decoy_pass = np.zeros((n_levels, n_floors), np.int64)
        floor_precision = np.zeros((SALTS, n_levels, n_floors))
        query_of_pair = np.searchsorted(np.array(chosen), pair_q)
        bin_of_query = bin_of[chosen]
        for salt in range(1, SALTS + 1):
            pa, gr, lv, summed = rows_for(mates, k, idx, salt=salt * 0x51ED27, bits=bits)
            assert len(pa) == len(pair)  # the same rows: sampling changes levels, not what is shared
            tr = gr == best[pair_q[pa]]
            rel = relative[pa]
            dp, _, dl, ds = rows_for(decoys, k, idx, salt=salt * 0x51ED27, bits=bits)
            for level in range(n_levels):
                score = scores(lv, summed, level, charge)
                decoy_score = scores(dl, ds, level, charge)
                top = top_of_pair(pa, score)
                hit_bits[level] += [summed[:, level][tr].sum(), lv[:, level][tr].sum()]
                for f, floor in enumerate(FLOORS):
                    on = score >= floor
                    cube[level, f] += np.bincount(cell[on], minlength=n_bins * N_BANDS)
                    for g, margin in enumerate(MARGINS):
                        kept = on & (score >= top - margin)
                        best_cube[g, level, f] += np.bincount(cell[kept], minlength=n_bins * N_BANDS)
                    floor_precision[salt - 1, level, f] = (on & tr).sum() / max(on.sum(), 1)
                    true_pairs = np.bincount(query_of_pair[pa[on & tr]], minlength=len(chosen))
                    np.add.at(query_pass[level, f], (bin_of_query, true_pairs), 1)
                    decoy_pass[level, f] += (decoy_score >= floor).sum()
            for level in range(LEVELS_MAX):
                for h in HITS:
                    on = lv[:, level] >= h
                    spread.setdefault(f"{level},{h}", []).append((on & tr).sum() / max(on.sum(), 1))
                    salted.setdefault(f"{level},{h}", []).append((
                        (on & tr & rel).sum() / max(relative.sum(), 1),
                        (on & ~tr & rel).sum() / max(relative.sum(), 1),
                        (on & tr).sum(),
                        on.sum(),
                    ))
        out["k"][str(k)] = {
            "postings": int(idx[3].size),
            "cells": table,
            "key_bits": [float(bits.mean()), *map(float, np.quantile(bits, [0.05, 0.95]))],
            "charge": charge,
            "floor_cube": cube.tolist(),
            "best_cube": best_cube.tolist(),
            "bits_per_true_hit": (hit_bits[:, 0] / np.maximum(hit_bits[:, 1], 1)).round(2).tolist(),
            "floor_query_pass": query_pass.tolist(),
            "floor_decoys": decoy_pass.tolist(),
            "floor_precision_sd": (floor_precision.std(0) / np.maximum(floor_precision.mean(0), 1e-9)).round(3).tolist(),
            "floor_rows": np.bincount(cell, minlength=n_bins * N_BANDS).tolist(),
            "true_by_query": per_query.tolist(),
            "decoy": [[int(m), int(n)] for m, n in zip(d_keys, d_counts, strict=True)],
            "real": real,
            "precision_sd": {key: float(np.std(v) / max(np.mean(v), 1e-9)) for key, v in spread.items()},
            "salted": {  # means over salts: recall, others, and hits on the true group and in all
                key: [float(x) for x in np.mean(v, 0)] for key, v in salted.items()
            },
        }
        Path(a.out).write_text(json.dumps(out))
        print(f"k{k}: {len(pair):,} rows, {len(table)} cells, {time.perf_counter() - t0:.0f} s, peak "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB", flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30 > MAX_RSS_GB:
            raise SystemExit(f"over {MAX_RSS_GB} GB")
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
