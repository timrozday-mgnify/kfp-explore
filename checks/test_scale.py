"""The scale profile's sampling and slices (measure/scale-profile/scale.py), on a local members table.

Run from anywhere, in kfp's venv: <kfp>/.venv/bin/python checks/test_scale.py <members.parquet>"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "measure/scale-profile"))
import scale  # noqa: E402

members = sys.argv[1]
ids = np.arange(1, 2_000_000)
u = scale.position(ids)
assert abs(u.mean() - 0.5) < 1e-3 and abs((u < 0.01).mean() - 0.01) < 1e-3  # uniform on [0, 1)

with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    (d / "samples").mkdir()
    a = argparse.Namespace
    scale.count(a(bucket=members, out=str(d / "count.json")))
    total = json.loads((d / "count.json").read_text())
    (d / "totals.json").write_text(json.dumps({"proteins": total["proteins"]}))
    scale.sample(a(bucket=members, totals=str(d / "totals.json"), top=30_000, queries=1000,
                   out=str(d / "samples/s.parquet"), queries_out=str(d / "q.parquet")))  # fmt: skip
    kept = pl.read_parquet(d / "samples/s.parquet")
    held = pl.read_parquet(d / "q.parquet")
    assert abs(kept.height / 30_000 - 1) < 0.05 and abs(held.height / 1000 - 1) < 0.2
    assert not set(kept["protein_id"].to_list()) & set(held["protein_id"].to_list())  # queries past every rung
    common = dict(samples=str(d / "samples"), totals=str(d / "totals.json"))
    small = dict(rung=3000.0, **common)
    large = dict(rung=30_000.0, **common)
    inner = set(pl.concat(list(scale.rung_tables(a(**small), ["protein_id"])))["protein_id"].to_list())
    outer = set(pl.concat(list(scale.rung_tables(a(**large), ["protein_id"])))["protein_id"].to_list())
    assert inner < outer  # nested
    scale.units(a(out=str(d / "units.json"), **large))
    units = json.loads((d / "units.json").read_text())
    assert units["proteins"] == len(outer) and 0.05 < units["s2"] < 0.07

    def run(kept_letters, k):
        scale.KEPT_LETTERS = kept_letters
        scale.kmers(a(out=str(d / "k.json"), k=k, units=str(d / "units.json"), **large))
        return json.loads((d / "k.json").read_text())

    for k in (5, 7, 9):
        whole, part = run(1e12, k), run(units["residues"] / 8, k)
        assert whole["slice"] == 1 and part["slice"] == 8
        print(f"k{k}: distinct {whole['distinct']:.4g} / {part['distinct']:.4g} at 1 in 8; decoys over composition "
              f"{whole['decoy_ratio']:.2f} / {part['decoy_ratio']:.2f}; between {whole['between_ratio']:.3g} / {part['between_ratio']:.3g}")  # fmt: skip
        assert abs(part["distinct"] / whole["distinct"] - 1) < 0.05
        assert abs(part["windows"] / whole["windows"] - 1) < 0.05
        if k <= 7:
            assert abs(part["decoy_ratio"] / whole["decoy_ratio"] - 1) < 0.5
    assert np.isclose(sum(whole["holders_log2"]), whole["kept_kmers"])
print("ok")
