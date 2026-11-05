"""Scale profile of a protein release (docs/plan.md, Scaling to full size): nested uniform samples,
sampled by a hash of protein_id so that each rung holds the one below, with statistics per rung
and whole-release anchors. Steps, run by main.nf:

- ``count``: one members bucket -> whole-release anchors (proteins, residues, units, singletons).
- ``sample``: one bucket -> the proteins of the largest rung, and the held-out query proteins.
- ``units``: rung -> proteins, residues, units, singletons, composition, lengths.
- ``kmers``: rung, k -> distinct k-mers, Good-Turing counts, holders, hot tail, decoys and matches
  between random halves, from a hash slice of the k-mers sized to memory.
- ``fasta``: rung -> FASTA for DIAMOND.
- ``spectrum``: rung, DIAMOND hits -> other groups per query by identity bin (the homologue spectrum).
- ``summary``: everything -> scale-profile.json, and check.tsv: each statistic's largest rung predicted
  from the others, whose error is the band on the page's curves.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web"))
import analyse  # noqa: E402

BATCH = 200_000  # sequences encoded at a time
KEPT_LETTERS = 2e8  # k-mers kept from a rung, by hash slice: about 10 bytes each while counting
MERGE_ROWS = 2e8
IDENTITY_EDGES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0001]
LENGTH_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def mix(x):
    """splitmix64, for protein ids and k-mers alike."""
    z = np.asarray(x).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def position(protein_id):
    """Each protein's place in [0, 1): a rung of N of T proteins holds those below N / T."""
    return mix(protein_id).astype(float) / 2.0**64


def count(a):
    t = pl.scan_parquet(a.bucket).select("cluster_rep", n=pl.col("sequence").str.len_chars()).collect(engine="streaming")
    sizes = t.group_by("cluster_rep").len()["len"]
    result = {
        "proteins": t.height, "residues": int(t["n"].sum()), "units": sizes.len(), "singletons": int((sizes == 1).sum()),
    }  # fmt: skip
    Path(a.out).write_text(json.dumps(result) + "\n")


def sample(a):
    t = pl.read_parquet(a.bucket, columns=["protein_id", "cluster_rep", "sequence"])
    t = t.with_columns(u=pl.Series(position(t["protein_id"].to_numpy())))
    total = json.loads(Path(a.totals).read_text())["proteins"]
    top = min(a.top / total, 1.0)
    held = a.queries / total
    t.filter(pl.col("u") < top).write_parquet(a.out)
    # Queries sit past every rung, so a rung never holds its own query, even on a small set.
    t.filter((pl.col("u") >= 1 - held)).write_parquet(a.queries_out)


def rung_files(a):
    return sorted(Path(a.samples).glob("*.parquet"))


def rung_share(a):
    return min(a.rung / json.loads(Path(a.totals).read_text())["proteins"], 1.0)


def rung_tables(a, columns):
    share = rung_share(a)
    for f in rung_files(a):
        yield pl.read_parquet(f, columns=[*columns, "u"]).filter(pl.col("u") < share)


def units(a):
    sizes, lengths, letters = [], [], np.zeros(256)
    for t in rung_tables(a, ["cluster_rep", "sequence"]):
        sizes.append(t["cluster_rep"])
        lengths.append(t["sequence"].str.len_chars().to_numpy())
        letters += np.bincount(np.frombuffer("".join(t["sequence"].to_list()).encode(), np.uint8), minlength=256)
    per_unit = pl.concat([s.to_frame() for s in sizes]).group_by("cluster_rep").len()["len"]
    n = np.concatenate(lengths)
    amino = letters[np.frombuffer(analyse.AMINO, np.uint8)]
    f = amino / max(amino.sum(), 1)
    result = {
        "rung": a.rung, "proteins": int(n.size), "residues": int(n.sum()), "units": per_unit.len(),
        "singletons": int((per_unit == 1).sum()), "mean_length": float(n.mean()), "mean_length2": float((n.astype(float) ** 2).mean()),
        "length_quantiles": np.quantile(n, LENGTH_QUANTILES).tolist(), "s2": float((f**2).sum()), "composition": f.tolist(),
    }  # fmt: skip
    Path(a.out).write_text(json.dumps(result) + "\n")


def kmers(a):
    """Windows and matches in a slice of the k-mer space, scaled up."""
    letters = json.loads(Path(a.units).read_text())
    s = max(int(np.ceil(np.log2(max(letters["residues"] / KEPT_LETTERS, 1)))), 0)
    mask = np.uint64(2**s - 1)
    chunks, held, windows = [], 0, np.zeros(3)

    def merge(frames):
        return pl.concat(frames).group_by("x").agg(pl.col("a").sum(), pl.col("b").sum(), pl.col("r").sum())

    for t in rung_tables(a, ["protein_id", "sequence"]):
        side_all = (mix(t["protein_id"].to_numpy()) & np.uint64(1)).astype(bool)
        seqs = t["sequence"].to_list()
        for start in range(0, len(seqs), BATCH):
            code = analyse.ENCODE[np.frombuffer("*".join(seqs[start : start + BATCH]).encode(), np.uint8)]
            x, at = analyse.kmers(code, a.k)
            owner = np.cumsum(code == analyse.SEPARATOR)[at]
            side = side_all[start : start + BATCH][owner]
            for name, values, on in (
                ("a", x, side), ("b", x, ~side), ("r", analyse.reversed_kmers(x, a.k), ~side),  # one half reversed
            ):  # fmt: skip
                v, w = values[on], owner[on]
                kept = (mix(v) & mask) == 0
                v, w = v[kept], w[kept]
                pairs = np.unique(np.stack([v, w], 1), axis=0)[:, 0] if len(v) else v  # per protein, once
                keys, n = np.unique(pairs, return_counts=True)
                windows["abr".index(name)] += len(pairs)
                zero = {c: np.zeros(len(keys), np.int32) for c in "abr"}
                chunks.append(pl.DataFrame({"x": keys, **zero, name: n.astype(np.int32)}))
                held += len(keys)
            if held > MERGE_ROWS:
                chunks = [merge(chunks)]
                held = chunks[0].height
    t = merge(chunks) if chunks else pl.DataFrame(schema={"x": pl.Int64, "a": pl.Int32, "b": pl.Int32, "r": pl.Int32})
    holders = (t["a"] + t["b"]).to_numpy().astype(np.int64)
    present = holders > 0
    h = holders[present]
    s2 = letters["s2"]
    scale = 2.0**s
    na, nb, nr = windows
    a_x, b_x, r_x = (t[c].to_numpy().astype(float) for c in "abr")
    expected = max(na, 1) * max(nb, 1) * scale * s2**a.k
    expected_r = max(na, 1) * max(nr, 1) * scale * s2**a.k
    order = np.sort(h)[::-1]
    decile = order[: max(len(order) // 100, 1)]
    result = {
        "rung": a.rung, "k": a.k, "slice": int(2**s), "kept_kmers": int(present.sum()),
        "distinct": float(present.sum() * scale), "windows": float(h.sum() * scale),
        "singletons": float((h == 1).sum() * scale), "doubletons": float((h == 2).sum() * scale),
        "unseen_share": float((h == 1).sum() / max(h.sum(), 1)),
        "top1_postings": float(decile.sum() / max(h.sum(), 1)), "max_holders": int(order[0]) if len(order) else 0,
        "holders_log2": np.bincount(np.log2(np.maximum(h, 1)).astype(int), minlength=1).tolist(),
        "between_ratio": float((a_x * b_x).sum() / expected), "decoy_ratio": float((a_x * r_x).sum() / expected_r),
        "decoy_matches": float((a_x * r_x).sum()),
    }  # fmt: skip
    Path(a.out).write_text(json.dumps(result) + "\n")


def fasta(a):
    with open(a.out, "w") as out:
        for t in rung_tables(a, ["protein_id", "sequence"]):
            for i, seq in zip(t["protein_id"].to_list(), t["sequence"].to_list(), strict=True):
                out.write(f">{i}\n{seq}\n")


def queries(a):
    t = pl.concat([pl.read_parquet(f, columns=["protein_id", "sequence"]) for f in sorted(Path(a.samples).glob("*.parquet"))])
    if t.height > a.count:
        t = t.sort("protein_id").sample(a.count, seed=1)
    with open(a.out, "w") as out:
        for i, seq in zip(t["protein_id"].to_list(), t["sequence"].to_list(), strict=True):
            out.write(f">{i}\n{seq}\n")


def spectrum(a):
    names = ["q", "s", "identity", "cover"]
    schema = {"q": pl.Int64, "s": pl.Int64, "identity": pl.Float64, "cover": pl.Float64}
    hits = pl.read_csv(a.hits, separator="\t", has_header=False, new_columns=names, schema_overrides=schema)
    group = pl.concat([t.select(s="protein_id", g="cluster_rep") for t in rung_tables(a, ["protein_id", "cluster_rep"])])
    n_queries = sum(1 for line in open(a.queries_fa) if line.startswith(">"))
    best = (
        hits.join(group, on="s").with_columns(x=pl.col("identity") / 100)
        .group_by("q", "g").agg(pl.col("x").max())
    )  # fmt: skip
    edges = np.array(IDENTITY_EDGES)
    x = best["x"].to_numpy()
    by_bin = np.histogram(x, bins=edges)[0]
    nearest = best.group_by("q").agg(pl.col("x").max())["x"].to_numpy()
    result = {
        "rung": a.rung, "queries": n_queries, "identity_edges": IDENTITY_EDGES,
        "groups_per_query": (by_bin / max(n_queries, 1)).tolist(),
        "nearest_share": (np.histogram(nearest, bins=edges)[0] / max(n_queries, 1)).tolist(),
        "no_relative": float(1 - len(nearest) / max(n_queries, 1)),
    }  # fmt: skip
    Path(a.out).write_text(json.dumps(result) + "\n")


def predicted(xs, ys):
    """The last value from the others: the last two's slope in log-log, as curves here flatten."""
    x, y = np.log(xs), np.log(np.maximum(ys, 1e-300))
    return float(np.exp(y[-2] + (y[-2] - y[-3]) / (x[-2] - x[-3]) * (x[-1] - x[-2])))


def summary(a):
    def read(pattern):
        return [json.loads(p.read_text()) for p in sorted(Path(a.dir).glob(pattern))]

    anchors = {key: sum(c[key] for c in read("count/*.json")) for key in ("proteins", "residues", "units", "singletons")}
    unit_rows = sorted(read("units/*.json"), key=lambda r: r["rung"])
    kmer_rows = sorted(read("kmers/*.json"), key=lambda r: (r["k"], r["rung"]))
    spectra = sorted(read("spectrum/*.json"), key=lambda r: r["rung"])
    out = {"source": a.source, "anchors": anchors, "rungs": unit_rows, "kmers": kmer_rows, "spectrum": spectra}
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    checks = []

    def check(name, rows, value):
        if len(rows) >= 3:
            xs = np.array([r["rung"] for r in rows], float)
            ys = np.array([value(r) for r in rows], float)
            guess = predicted(xs, ys)
            checks.append({"statistic": name, "largest_rung": rows[-1]["rung"], "actual": ys[-1], "predicted": guess,
                           "ratio": guess / ys[-1] if ys[-1] else None})  # fmt: skip

    check("units", unit_rows, lambda r: r["units"])
    check("singleton_share", unit_rows, lambda r: r["singletons"] / max(r["units"], 1))
    for k in sorted({r["k"] for r in kmer_rows}):
        rows = [r for r in kmer_rows if r["k"] == k]
        check(f"distinct_k{k}", rows, lambda r: r["distinct"])
        check(f"between_ratio_k{k}", rows, lambda r: r["between_ratio"])
        check(f"decoy_ratio_k{k}", rows, lambda r: r["decoy_ratio"])
    if spectra:
        for b, edge in enumerate(IDENTITY_EDGES[:-1]):
            check(f"groups_from_{edge}", spectra, lambda r, b=b: r["groups_per_query"][b])
    pl.DataFrame(checks, infer_schema_length=None).write_csv(a.check, separator="\t")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("count", count), ("sample", sample), ("units", units), ("kmers", kmers), ("fasta", fasta),
                       ("queries", queries), ("spectrum", spectrum), ("summary", summary)):  # fmt: skip
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--out", required=True)
        if name in ("count", "sample"):
            p.add_argument("--bucket", required=True)
        if name == "sample":
            p.add_argument("--totals", required=True)
            p.add_argument("--top", type=float, required=True, help="proteins in the largest rung")
            p.add_argument("--queries", type=float, required=True, help="proteins held out as candidate queries")
            p.add_argument("--queries-out", required=True)
        if name in ("units", "kmers", "fasta", "spectrum"):
            p.add_argument("--samples", required=True)
            p.add_argument("--totals", required=True)
            p.add_argument("--rung", type=float, required=True)
        if name == "kmers":
            p.add_argument("--k", type=int, required=True)
            p.add_argument("--units", required=True)
        if name == "queries":
            p.add_argument("--samples", required=True)
            p.add_argument("--count", type=int, required=True)
        if name == "spectrum":
            p.add_argument("--hits", required=True)
            p.add_argument("--queries-fa", required=True)
        if name == "summary":
            p.add_argument("--dir", required=True)
            p.add_argument("--check", required=True)
            p.add_argument("--source", required=True)
    a = parser.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
