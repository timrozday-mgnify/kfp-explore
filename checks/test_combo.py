"""Combination k-mers (docs/plan.md, phase 9): the model against simulation and kfp's measurements.

Run: uv run --no-project --with numpy python checks/test_combo.py"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "web"))
import model  # noqa: E402

t = json.loads((ROOT / "checks/data/sim-combo.json").read_text())
ids = model.COMBO_IDENTITIES
mid = (np.array(t["bins"][:-1]) + np.array(t["bins"][1:])) / 2
mid[-1] = 0.99

# Feature counts per start are kfp's: 11 and 9 pairs, 36 triples, of 3- and 4-mers within 16 letters.
assert [len(model.combo_sets(n, k, 16 - k)) for k, n in ((3, 2), (4, 2), (3, 3))] == [11, 9, 36]

# A single k-mer through the counts agrees with the chain.
gap = model.combo_recall(50, 9, 1, 0, 0, 8.0, ceiling=1) - model.recall(50, 9, 8.0, ids, ceiling=1)
print(f"single k-mer, counts against the chain: RMSE {np.sqrt(np.mean(gap**2)):.3f}")
assert np.sqrt(np.mean(gap**2)) < 0.03

errors = {1: [], 2: [], 3: []}
print("recall against simulation, no ceiling (RMSE, bias; m 1 to 128):")
for row in t["rows"]:
    k, n, lo, hi = row["k"], row["n"], row["lo"], row["hi"]
    e = []
    for level in range(8):
        pred = np.interp(mid, ids, model.combo_recall(50, k, n, lo, hi, float(2**level), ceiling=1))
        e.append(pred - np.array(row["recall"][level]))
    e = np.array(e)
    errors[n].append(e.ravel())
    print(f"  k{k} n{n} window {lo}-{hi}: {np.sqrt(np.mean(e**2)):.3f}, {e.mean():+.3f}")
for n, e in errors.items():
    rmse = float(np.sqrt(np.mean(np.concatenate(e) ** 2)))
    print(f"n {n}: RMSE {rmse:.3f}")
    assert rmse < 0.08, n

# Simulated reads against kfp's mode table: two shared features on one read, at identity 0.6-0.7 and 0.5-0.6.
KFP = {(5, 1): (0.92, 0.70), (6, 1): (0.82, 0.58), (7, 1): (0.69, 0.44), (8, 1): (0.55, 0.39),
       (3, 2): (0.97, 0.82), (4, 2): (0.82, 0.61), (3, 3): (0.85, 0.57)}  # fmt: skip
gaps = []
for row in t["rows"]:
    if (row["k"], row["n"]) in KFP and row["n"] == 1 or (row["k"], row["n"]) in KFP and row["lo"] == row["k"] and row["hi"] == 16 - row["k"]:
        sim = (row["two_on_a_read"][0][1], row["two_on_a_read"][0][0])
        ref = KFP[(row["k"], row["n"])]
        gaps += [sim[0] - ref[0], sim[1] - ref[1]]
        print(f"  k{row['k']} n{row['n']}: simulated {sim[0]:.2f} / {sim[1]:.2f}, kfp {ref[0]:.2f} / {ref[1]:.2f}")
print(f"against kfp: RMSE {np.sqrt(np.mean(np.square(gaps))):.3f}, mean {np.mean(gaps):+.3f}")
assert np.sqrt(np.mean(np.square(gaps))) < 0.1

# Chance from composition against decoys, where counts allow.
ratios = []
for row in t["rows"]:
    if row["chance_matches"] >= 100 and row["n"] > 1:
        pred = model.combo_chance(50, row["k"], row["n"], row["lo"], row["hi"], True, 1, t["residues"], model.S2)
        ratio = row["chance_per_pair_m1"] / pred
        print(f"  k{row['k']} n{row['n']} window {row['lo']}-{row['hi']}: decoys over composition {ratio:.2f}, from {row['chance_matches']}")
        ratios.append(ratio)
print(f"decoys over composition: {min(ratios):.2f} to {max(ratios):.2f}")
assert 0.8 < min(ratios) and max(ratios) < 4

# The panel, and an unavailable setting.
base = {"k": 3, "m": 8, "length": 50, "paired": True, "identity": 0.9, "alphabet": "protein", "combo_n": 2, "combo_lo": 3, "combo_hi": 13}
q = model.combo_panel(base, 0.0, model.S2, 1e9)
assert q["available"] and q["g"] == 11 and q["combo"][-1] > 0.9 and len(q["by_m_combo"]) == 14
assert not model.combo_panel({**base, "combo_lo": 20}, 0.0, model.S2, 1e9)["available"]
wide = model.combo_panel({**base, "combo_hi": 26}, 0.0, model.S2, 1e9)
assert wide["g"] == 24 and wide["chance"] > q["chance"] * (24 / 11) ** 1.5  # chance outgrows postings
print("ok")
