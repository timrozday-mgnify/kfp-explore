"""Phase 6's tables against kfp's matched comparisons (docs/plan.md; kfp's docs/tree-index-plan.md).

Run: uv run --no-project --with numpy python checks/test_policy.py"""

import json
from pathlib import Path

import numpy as np

t = json.loads((Path(__file__).parent.parent / "web/data/policy-mini-slice.json").read_text())
bins = t["identity_bins"]
BANDS, FLOOR = len(t["bands"]), t["floors"].index(40)
KFP_ID = [0.98, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]  # kfp's columns 1.0, 0.9 ... 0.6, as lower bin edges here
KFP_RETENTION = {  # share of a group's 32 keys a held-out protein holds
    "union": [0.25, 0.21, 0.20, 0.13, 0.13, 0.10, 0.08],
    "share>=0.25": [0.63, 0.52, 0.45, 0.32, 0.27, 0.19, 0.15],
    "nearest": [0.77, 0.60, 0.48, 0.35, 0.26, 0.21, 0.15],
    "frequency": [0.85, 0.72, 0.62, 0.48, 0.38, 0.28, 0.23],
}
KFP_LAYER = {  # window 8, (radius, policy): postings over S3, less the base's 1; mean recall at 80, 75, 70%
    ("tree:0.57", "medoid"): (4.76, 0.783),
    ("tree:0.57", "longest"): (4.80, 0.777),
    ("tree:0.57", "central"): (3.86, 0.717),
    ("tree:0.57", "composite:0"): (5.71, 0.790),
    ("tree:0.57", "composite:0.25"): (4.83, 0.793),
    ("tree:0.57", "composite:0.5"): (4.37, 0.770),
    ("tree:0.57", "composite:0.75"): (3.68, 0.693),
    ("units", "composite:0.25"): (4.35, 0.777),
    ("tree:0.83", "composite:0.25"): (4.76, 0.793),
    ("tree:0.34", "composite:0.25"): (5.49, 0.813),
}


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    return float(np.corrcoef(ranks(np.array(a)), ranks(np.array(b)))[0, 1])


retained = {name: {bins[b]: r for b, r in rows} for name, rows in t["retention"]["by_policy"].items()}
errors = []
for name, values in KFP_RETENTION.items():
    errors += [retained[name][x] - v for x, v in zip(KFP_ID, values, strict=True)]
rmse = float(np.sqrt(np.mean(np.square(errors))))
for x in bins[2:-1]:
    assert retained["frequency"][x] > retained["share>=0.25"][x] > retained["union"][x], x
assert rmse < 0.1, rmse

mine = {(c["radius"], c["policy"]): c for c in t["configs"] if c["k"] == 6 and c["window"] == 8 and not c["budget"]}
keys = list(KFP_LAYER)
low = [bins.index(x) for x in (0.7, 0.75, 0.8)]
recall = [np.mean([mine[k]["called_true"][FLOOR][b] / t["queries_by_bin"][b] for b in low]) for k in keys]
postings = [mine[k]["postings"] for k in keys]
kfp_postings, kfp_recall = zip(*(KFP_LAYER[k] for k in keys), strict=True)
s_recall, s_postings = spearman(recall, kfp_recall), spearman(postings, kfp_postings)
assert s_recall > 0.5, s_recall
print(f"retention RMSE {rmse:.3f}; recall rank correlation with kfp {s_recall:.2f}, postings {s_postings:.2f}")
ratio = np.array(postings) / np.array(kfp_postings)
print("postings over kfp's, relative to composite 0.25:")
for k, r in zip(keys, ratio / ratio[keys.index(("tree:0.57", "composite:0.25"))], strict=True):
    print(f"  {k[0]} {k[1]}: {r:.2f}")
