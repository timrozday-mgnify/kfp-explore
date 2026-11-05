"""Summarise kfp's `tree-index-benchmark` results for the page: real tree and flat tiers, scored on
one set of strains. Writes a small JSON table; the page reads it and computes nothing heavy.

Run: uv run --no-project --with numpy --with polars python measure/benchmark_summary.py \
    <extracted results/> web/data/kfp-benchmark.json
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

TIERS = ("slim", "standard", "large")
LEVELS = ("0.99", "0.95", "0.9", "0.8", "mgnify90", "pfam")
IDENTITY_LEVELS = LEVELS[:4]
PRECISION_GRID = np.linspace(0, 1, 201)
TAIL_BITS = list(range(20, 125, 5))


def recall_at_precision(levels, level):
    """The most recall at any cut of the ranking with precision >= each grid value (0 if none)."""
    t = levels.filter(pl.col("level") == level)
    recall, precision = t["recall"].to_numpy(), t["precision"].to_numpy()
    order = np.argsort(precision)
    best = np.maximum.accumulate(recall[order][::-1])[::-1]  # best recall at this precision or above
    at = np.searchsorted(precision[order], PRECISION_GRID - 1e-12)
    return [round(float(best[i]), 4) if i < len(best) else 0.0 for i in at]


def summarise_run(path, name):
    score = json.loads((path / "score.json").read_text())
    levels = pl.read_parquet(path / "levels.parquet")
    curves = pl.read_csv(path / "curves.tsv", separator="\t", infer_schema_length=None)
    by_id = curves.filter((pl.col("side") == "recall") & (pl.col("by") == "identity_bin"))
    method, tier = name.removesuffix("-tier2").split("-")
    return {
        "method": "tree tier 2" if name.endswith("tier2") else f"{method} tier 1",
        "tier": tier,
        "index_gb": score["index_bytes"] / 1e9,
        "query_s": score["query_s"],
        "units": score["units"],
        "emd": score["emd"],
        "whole": {x: [score[f"recall_{x}"], score[f"precision_{x}"]] for x in LEVELS},
        "at_precision": {x: recall_at_precision(levels, x) for x in LEVELS},
        "by_identity": {x: [float(v) for v in by_id[x]] for x in IDENTITY_LEVELS},
        "identity_bins": [float(v) for v in by_id["bin"]],
    }


def main(results, out):
    results = Path(results)
    runs = {
        f"{m}-{t}{s}": None
        for m, s in (("flat", ""), ("tree", ""), ("tree", "-tier2"))
        for t in TIERS
    }
    runs = {name: summarise_run(results / "nearest" / name, name) for name in runs}
    bench = pl.read_csv(results / "bench.tsv", separator="\t", infer_schema_length=None)
    tiers, tail = {}, {}
    for row in bench.iter_rows(named=True):
        tiers[row["name"]] = {k: row[k] for k in ("postings", "bytes", "layers")}
        query = json.loads((results / f"runs/q-{row['name']}/query.json").read_text())
        evidence = pl.read_parquet(results / f"runs/q-{row['name']}/evidence.parquet")
        chance = dict(zip(evidence["bits"], evidence["chance"], strict=True))
        pairs = query["read_pairs"]
        tail[row["name"]] = [chance.get(float(b), 0.0) / pairs for b in TAIL_BITS]
    postings = [tiers[t]["postings"] for t in TIERS]
    slope, fixed = np.polyfit(postings, [tiers[t]["bytes"] for t in TIERS], 1)
    with open(results / "gate.tsv", newline="") as f:
        gate = [r for r in csv.DictReader(f, delimiter="\t") if r["measure"] in LEVELS]
    table = {
        "source": "kfp tree-index-benchmark, HPC run of 2026-11-05 (1-in-100 subset)",
        "strains": json.loads((results / "nearest/flat-slim/score.json").read_text())["strains"],
        "read_pairs": pairs,
        "levels": LEVELS,
        "precision_step": float(PRECISION_GRID[1]),
        "runs": runs,
        "tiers": tiers,
        "tail_bits": TAIL_BITS,
        "tail_per_pair": tail,
        "size_fit": {"fixed_gb": fixed / 1e9, "bytes_per_posting": slope},
        "gate": [
            {"tree": r["tree"], "level": r["measure"], "value": float(r["tree_value"]),
             "precision": float(r["tier1_precision"])}
            for r in gate
        ],
    }  # fmt: skip
    Path(out).write_text(json.dumps(table, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
