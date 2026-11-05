"""Tables for grouping and key policy (docs/plan.md, phase 6), on keys built kfp's way.

Proteins are held out before the build. The rest are grouped by kfp's ``build.layer_groups``, and keys
picked by its ``window_kmers`` (the rarest k-mer per window), ``on_reps`` (representative or soft
intersection) and ``allocate`` (a key budget). Read pairs and reversed decoys from the held-out
proteins are scored with kfp's evidence on each configuration. A read's true group holds the query's
nearest relative, so relatives within the radius count as correct.

Run from kfp's checkout, one job with capped threads:
RAYON_NUM_THREADS=2 POLARS_MAX_THREADS=2 uv run --no-sync python <this file> data/mini-slice/members.parquet --out x.json
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
from kmer_functional_profiler import _core

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.cwd() / "workflows/tree-index"))
import sim_grid as sg  # noqa: E402

import build  # noqa: E402
import survey  # noqa: E402

BINS = sg.BINS
NO_RELATIVE = len(BINS) - 1
FLOORS = (0, 20, 30, 40, 50, 60, 80, 120)
BANDS = ("true", "x>=0.9", "x>=0.7", "x>=0.5", "other")
X_EDGES = np.array([0.9, 0.7, 0.5])
MARGIN = 3  # kfp's best-group rule
FRAMES = 6  # kfp's charge counts every frame's windows
MAX_RSS_GB = 12  # of the laptop's 24
RETENTION_N = 32
RETENTION_MIN_MEMBERS = 8

# (k, window, radius, policy, budget): kfp's matched comparisons, then its tiers.
CONFIGS = [
    *[(6, 8, "tree:0.57", p, None) for p in (
        "medoid", "longest", "central", "composite:0", "composite:0.25", "composite:0.5", "composite:0.75")],
    *[(6, 8, r, "composite:0.25", None) for r in ("units", "tree:0.34", "tree:0.83")],
    (6, 8, "none", "longest", None),
    *[(6, w, "tree:0.57", "composite:0.25", None) for w in (4, 16, 32, 64)],
    *[(6, w, "tree:0.57", p, b) for w, b in ((16, "sqrt:2.6"), (8, "sqrt:14"), (8, "uniform:14"))
      for p in ("composite:0", "composite:0.25", "frequency")],
    *[(k, 8, "tree:0.57", "composite:0.25", None) for k in (5, 7, 8)],
    *[(k, 8, "none", "longest", None) for k in (5, 7, 8)],
    (5, 16, "tree:0.57", "composite:0.25", "sqrt:2.6"),
    (5, 8, "tree:0.57", "composite:0.25", "sqrt:14"),
]


def watchdog():
    """Stop the whole process over the memory limit: a local run once crashed the laptop."""
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30 > MAX_RSS_GB:
            print(f"over {MAX_RSS_GB} GB, stopping", flush=True)
            os._exit(3)
        time.sleep(1)


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def hold_out(members, a):
    """(query rows, kept rows): whole held units, so divergent queries, and members of kept units."""
    lengths = pl.col("sequence").str.len_chars()
    span = lengths.is_between(sg.MATE * 2 + sg.GAP, 1000)
    is_rep = pl.col("protein_id") == pl.col("cluster_rep")
    held = members.filter(is_rep, span).sample(a.held, seed=a.seed)
    kept = members.join(held.select("cluster_rep"), on="cluster_rep", how="anti")
    picked = kept.filter(~is_rep, span).sample(a.member_queries, seed=a.seed)
    return pl.concat([held, picked]), kept.join(picked.select("protein_id"), on="protein_id", how="anti")


def read_pairs(query, chosen, n_pairs, rng):
    pair_q, mates, decoys = [], [], []
    for q in chosen:
        s = query[q]
        for _ in range(n_pairs):
            gap = int(rng.integers(0, sg.GAP + 1))
            start = int(rng.integers(0, len(s) - 2 * sg.MATE - gap + 1))
            m1, m2 = s[start : start + sg.MATE], s[start + sg.MATE + gap : start + 2 * sg.MATE + gap]
            pair_q.append(q)
            mates += [m1, m2]
            decoys += [m1[::-1], m2[::-1]]
    return np.array(pair_q), mates, decoys


def hashed_kmers(seqs, k, owner_of, log_f):
    """Distinct (owner, hash, composition bits) over ``seqs``, hashed as kfp's index."""
    ks, owners = [], []
    for i, s in enumerate(seqs):
        ks += [s[j : j + k] for j in range(len(s) - k + 1)]
        owners += [owner_of[i]] * (len(s) - k + 1)
    got = _core.hash_proteins(ks, k, alphabet="protein")
    at = np.asarray(got["seq"])
    bits = np.array([-log_f[np.frombuffer(ks[i], np.uint8)].sum() for i in at])
    return pl.DataFrame({"owner": np.array(owners)[at], "h": np.asarray(got["hash"]), "q0": bits}).unique(["owner", "h"])


def thin_by_share(start_group, pos, h, budget):
    """Union on a representative, then ``allocate`` keeps each group's most shared keys."""
    df = pl.DataFrame({"h": h, "g": start_group[pos]})
    share = df.group_by("g", "h").agg(c=pl.len())
    group = start_group.copy()
    u_pos, u_h = build.on_reps(group, pos, h, "composite:0")
    rep_of_first = pl.DataFrame({"g": start_group, "rep": group}).unique("g")
    table = (
        pl.DataFrame({"pos": u_pos, "h": u_h})
        .join(rep_of_first.rename({"rep": "pos"}), on="pos")
        .join(share, on=["g", "h"])
        .sort(["c", "h"], descending=[True, False])
        .with_row_index("rank")
    )
    # allocate keeps the least values per unit, so the rank stands in for the hash.
    kept_pos, kept_rank = build.allocate(
        group, table["pos"].to_numpy().astype(np.int64), table["rank"].to_numpy().astype(np.uint64), budget
    )
    return kept_pos, table["h"].to_numpy()[kept_rank.astype(np.int64)], group


def keys_for(config, groups, windows):
    """(position of each key's group, hash, each position's group), from kfp's functions."""
    k, w, radius, policy, budget = config
    pos, h = windows(k, w)
    start = groups(radius)
    if policy == "frequency":
        return thin_by_share(start, pos, h, budget)
    group = start.copy()
    pos, h = build.on_reps(group, pos, h, policy)
    if budget:
        pos, h = build.allocate(group, pos, h, budget)
    return pos, h, group


def score_rows(kmers, key_table, charge):
    """(owner, group, score) of kfp's evidence: key bits = composition less log2 of groups holding it."""
    joined = kmers.join(key_table, on="h").with_columns(
        bits=(pl.col("q0") - pl.col("holders").log(2)).clip(lower_bound=0)
    )
    return (
        joined.group_by("owner", "g")
        .agg(hits=pl.len(), s=pl.col("bits").sum())
        .with_columns(score=pl.col("s") - (pl.col("hits") - 1) * charge)
    )


def evaluate(pos, h, group, kmers, decoy_kmers, truth, k):
    """Per floor and identity bin: rows in each band, for every group over the floor and for each
    read's best groups; true targets and false groups called by any read pair; decoys."""
    keys = pl.DataFrame({"g": pos, "h": h}).unique()
    holders = keys.group_by("h").agg(holders=pl.len())
    table = keys.join(holders, on="h")
    charge = float(np.log2(2 * FRAMES * (sg.MATE - k + 1)))
    rows = score_rows(kmers, table, charge)
    decoys = score_rows(decoy_kmers, table, charge)
    pair_query, pair_bin, true_group, aligned = truth
    rel = (
        pl.DataFrame({"query": aligned[0], "pos": aligned[1], "x": aligned[2]})
        .with_columns(g=pl.Series(group[aligned[1]]))
        .group_by("query", "g")
        .agg(x=pl.col("x").max())
    )
    rows = (
        rows.with_columns(query=pl.Series(pair_query[rows["owner"].to_numpy()]))
        .join(rel, on=["query", "g"], how="left")
        .with_columns(x=pl.col("x").fill_null(0.0))
    )
    p = rows["owner"].to_numpy()
    g = rows["g"].to_numpy()
    x = rows["x"].to_numpy()
    tg = true_group[pair_query[p]]
    t = (g == group[np.maximum(tg, 0)]) & (tg >= 0)
    band = np.where(t, 0, np.select([x >= e for e in X_EDGES], [1, 2, 3], 4))
    score = rows["score"].to_numpy()
    b = pair_bin[p]
    top = pl.DataFrame({"p": p, "s": score}).with_columns(t=pl.col("s").max().over("p"))["t"].to_numpy()
    n_bins, n_bands = NO_RELATIVE + 1, len(BANDS)
    cube, best, called_true, called_false, decoy, sticky = [], [], [], [], [], []
    q = pair_query[p]
    for floor in FLOORS:
        on = score >= floor
        cube.append(np.bincount(b[on] * n_bands + band[on], minlength=n_bins * n_bands).tolist())
        kept = on & (score >= top - MARGIN)
        best.append(np.bincount(b[kept] * n_bands + band[kept], minlength=n_bins * n_bands).tolist())
        per_q = pl.DataFrame({"q": q[on], "g": g[on], "t": t[on], "b": b[on]}).unique(["q", "g"])
        called_true.append(np.bincount(per_q.filter(pl.col("t"))["b"].to_numpy(), minlength=n_bins).tolist())
        called_false.append(np.bincount(per_q.filter(~pl.col("t"))["b"].to_numpy(), minlength=n_bins).tolist())
        over = decoys.filter(pl.col("score") >= floor)
        decoy.append(over.height)
        counts = np.sort(over.group_by("g").len()["len"].to_numpy())[::-1]
        sticky.append(float(counts[: max(len(counts) // 100, 1)].sum() / max(counts.sum(), 1)))
    return {
        "groups": keys["g"].n_unique(),
        "postings": keys.height,
        "cube": cube,
        "best": best,
        "called_true": called_true,
        "called_false": called_false,
        "decoys": decoy,
        "decoy_top1pct_share": sticky,
    }


def retention(group, relative_pos, chosen_bins, chosen, query_kmers, k, windows):
    """kfp's table: of a group's n keys, the share a held-out protein holds, in groups of at least
    8 members, for four ways of choosing n keys. Least hash stands for random thinning."""
    pos, h = windows(k, 1)
    start = group.copy()
    members = np.bincount(start, minlength=len(start))
    fixed = f"fixed:{RETENTION_N}"
    built = {}
    g0 = start.copy()
    u_pos, u_h = build.on_reps(g0, pos, h, "composite:0")
    built["union"] = (*build.allocate(g0, u_pos, u_h, fixed), g0)
    g1 = start.copy()
    s_pos, s_h = build.on_reps(g1, pos, h, "composite:0.25")
    built["share>=0.25"] = (*build.allocate(g1, s_pos, s_h, fixed), g1)
    built["frequency"] = thin_by_share(start, pos, h, fixed)
    near = pl.DataFrame({"pos": pos, "h": h}).sort("pos", "h").group_by("pos", maintain_order=True).head(RETENTION_N)
    out = {}
    chosen = np.asarray(chosen)
    chosen = chosen[relative_pos[chosen] >= 0]
    queries = chosen[members[start[relative_pos[chosen]]] >= RETENTION_MIN_MEMBERS]
    for name, (kp, kh, grp) in built.items():
        keys = pl.DataFrame({"g": kp, "h": kh}).unique()
        size = keys.group_by("g").agg(n=pl.len())
        q_groups = pl.DataFrame({"owner": queries, "g": grp[relative_pos[queries]], "b": chosen_bins[queries]})
        held = query_kmers.select("owner", "h").join(q_groups, on="owner").join(keys, on=["g", "h"])
        hit = held.group_by("owner").agg(hit=pl.len())
        joined = q_groups.join(size, on="g").join(hit, on="owner", how="left").with_columns(pl.col("hit").fill_null(0))
        out[name] = joined.group_by("b").agg((pl.col("hit") / pl.col("n")).mean().alias("r")).sort("b").rows()
    nearest_keys = near.with_columns(pl.col("pos").alias("relpos")).select("relpos", "h")
    size = nearest_keys.group_by("relpos").agg(n=pl.len())
    q_rel = pl.DataFrame({"owner": queries, "relpos": relative_pos[queries], "b": chosen_bins[queries]})
    hit = query_kmers.select("owner", "h").join(q_rel, on="owner").join(nearest_keys, on=["relpos", "h"]).group_by("owner").agg(hit=pl.len())
    joined = q_rel.join(size, on="relpos").join(hit, on="owner", how="left").with_columns(pl.col("hit").fill_null(0))
    out["nearest"] = joined.group_by("b").agg((pl.col("hit") / pl.col("n")).mean().alias("r")).sort("b").rows()
    return {"counts": np.bincount(chosen_bins[queries], minlength=NO_RELATIVE + 1).tolist(), "by_policy": out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("members")
    p.add_argument("--out", required=True)
    p.add_argument("--held", type=int, default=3000)
    p.add_argument("--member-queries", type=int, default=1000)
    p.add_argument("--per-bin", type=int, default=100)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--only", type=int, help="the first N configurations, for a cost check")
    a = p.parse_args()
    threading.Thread(target=watchdog, daemon=True).start()
    rng = np.random.default_rng(a.seed)
    members = pl.read_parquet(a.members, columns=["protein_id", "cluster_rep", "sequence", "full_length"])
    query_rows, kept = hold_out(members, a)
    del members
    print(f"{query_rows.height} held out, {kept.height} kept", flush=True)
    query = [s.encode() for s in query_rows["sequence"]]
    seqs = [s.encode() for s in kept["sequence"]]

    t0 = time.perf_counter()
    qs, ps = sg.candidates(query, seqs, a.top)
    short = np.array([len(seqs[i]) <= 3000 for i in ps])
    qs, ps = qs[short], ps[short]
    got = _core.align_identity([query[q] for q in qs], [seqs[i] for i in ps])
    identity = np.asarray(got["identity"], float)
    cover = np.asarray(got["columns"], float) / np.array([len(query[q]) for q in qs])
    print(f"aligned {len(qs):,} pairs, {time.perf_counter() - t0:.0f} s, peak {peak():.1f} GB", flush=True)

    t0 = time.perf_counter()
    tree_args = argparse.Namespace(
        k=11, t_cap=0.001, levels=8, topology_t_base=0.02, topology_n_min=16, leaves="proteins",
        topology_tree="spanning", topology_k=8, topology_t=1.0, topology_max_holders=0,
    )  # fmt: skip
    tree = survey.Tree(kept, tree_args)
    n = tree.forest.n
    pos_of_leaf = np.empty(n, dtype=np.int64)
    pos_of_leaf[tree.lay["order"]] = np.arange(n)
    has_leaf = tree.leaf_of >= 0
    pos_of_seq = np.where(has_leaf, pos_of_leaf[np.maximum(tree.leaf_of, 0)], -1)
    print(f"tree: {n} leaves, {time.perf_counter() - t0:.0f} s, peak {peak():.1f} GB", flush=True)
    keep = pos_of_seq[ps] >= 0
    qs, ps, identity, cover = qs[keep], ps[keep], identity[keep], cover[keep]

    # Each query's nearest relative at cover 0.8, and the identity bin it falls in.
    relative_pos, best_x = np.full(len(query), -1), np.zeros(len(query))
    for i in np.flatnonzero(cover >= 0.8):
        if identity[i] > best_x[qs[i]]:
            best_x[qs[i]], relative_pos[qs[i]] = identity[i], pos_of_seq[ps[i]]
    bin_of = np.digitize(best_x, BINS) - 1
    bin_of[relative_pos < 0] = NO_RELATIVE
    bin_of[(relative_pos >= 0) & (bin_of < 0)] = NO_RELATIVE
    relative_pos[bin_of == NO_RELATIVE] = -1
    chosen = sorted(
        i for b in range(NO_RELATIVE + 1)
        for i in rng.permutation(np.flatnonzero(bin_of == b))[: a.per_bin].tolist()
    )  # fmt: skip
    pair_q, mates, decoy_mates = read_pairs(query, chosen, a.pairs, rng)
    pair_bin = bin_of[pair_q]
    print(f"{len(pair_q)} read pairs; by bin {np.bincount(pair_bin, minlength=NO_RELATIVE + 1).tolist()}", flush=True)
    mate_owner = np.repeat(np.arange(len(pair_q)), 2)
    letters = np.bincount(np.frombuffer(b"".join(seqs), np.uint8), minlength=256).astype(float)
    log_f = np.log2(np.maximum(letters / letters.sum(), 1e-300))
    aligned = (qs, pos_of_seq[ps], identity)
    truth = (pair_q, pair_bin, relative_pos, aligned)

    reps = [tree.seqs[i] for i in tree.rep_by_pos]
    groups_memo, windows_memo, read_memo = {}, {}, {}

    def groups(radius):
        if radius not in groups_memo:
            groups_memo[radius] = (
                np.arange(n, dtype=np.int64) if radius == "none" else build.layer_groups(tree, kept, radius)
            )
        return groups_memo[radius]

    def windows(k, w):
        if (k, w) not in windows_memo:
            vc = build.kmer_info(reps, k, log_f_for_kfp, "protein").group_by("hash").len().sort("hash")
            rank = (vc["hash"].to_numpy(), vc["len"].to_numpy().astype(np.uint64))
            windows_memo[(k, w)] = build.window_kmers(reps, k, w, 1, "protein", rank)
        return windows_memo[(k, w)]

    log_f_for_kfp = build.residue_log_freq(tree.seqs)

    def reads(k):
        if k not in read_memo:
            read_memo[k] = (hashed_kmers(mates, k, mate_owner, log_f), hashed_kmers(decoy_mates, k, mate_owner, log_f))
        return read_memo[k]

    out = {
        "source": a.members, "proteins": len(seqs), "leaves": int(n), "pairs": len(pair_q), "pairs_per_query": a.pairs,
        "mate": sg.MATE, "seed": a.seed, "identity_bins": BINS.tolist(), "floors": list(FLOORS), "bands": list(BANDS),
        "margin": MARGIN, "pairs_by_bin": np.bincount(pair_bin, minlength=NO_RELATIVE + 1).tolist(),
        "queries_by_bin": np.bincount(bin_of[chosen], minlength=NO_RELATIVE + 1).tolist(), "configs": [],
    }  # fmt: skip
    for config in CONFIGS[: a.only]:
        t0 = time.perf_counter()
        pos, h, group = keys_for(config, groups, windows)
        kmers, decoy_kmers = reads(config[0])
        row = evaluate(pos, h, group, kmers, decoy_kmers, truth, config[0])
        row |= dict(zip(("k", "window", "radius", "policy", "budget"), config, strict=True))
        out["configs"].append(row)
        Path(a.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
        print(f"{config}: {row['postings']:,} postings, {row['groups']:,} groups, "
              f"{time.perf_counter() - t0:.0f} s, peak {peak():.1f} GB", flush=True)  # fmt: skip

    t0 = time.perf_counter()
    protein_kmers = hashed_kmers([query[i] for i in chosen], 5, np.array(chosen), log_f)
    out["retention"] = {
        "k": 5, "n": RETENTION_N, "min_members": RETENTION_MIN_MEMBERS,
        **retention(groups("tree:0.57"), relative_pos, bin_of, chosen, protein_kmers, 5, windows),
    }  # fmt: skip
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"retention: {time.perf_counter() - t0:.0f} s, peak {peak():.1f} GB; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
