"""On-demand precision (web/live.py) against the tables, on different queries, so as a held-out check.

Run: uv run --no-project --with numpy python checks/test_live.py"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "web"))
import live  # noqa: E402
import model  # noqa: E402

COMMON = 1.0  # others per pair above which a setting is compared closely
live.load("mini-slice", (ROOT / "web/data/live-mini-slice.npz").read_bytes())
model.load_table("mini-slice", (ROOT / "web/data/tables-mini-slice.json").read_text())
tables = model.TABLES["mini-slice"]
BASE = {"index": "mini-slice", "alphabet": "protein", "length": 50, "paired": True, "floor": 3, "margin": -1, "truth": 2,
        "mix": "natural", "no_relative": 0.05, "coverage": 10, "targets": 1e4}  # fmt: skip


def request(**kw):
    return json.dumps({**BASE, **kw})


def panel(**kw):
    r = request(**kw)
    if not live.have(r):
        live.compute(r)
    return json.loads(live.panel(r))


errors = {"recall": [], "others": []}
for k, m in [(5, 8), (7, 8), (7, 32), (9, 8), (9, 32), (11, 8)]:
    r = request(k=k, m=m)
    live.compute(r)
    live.panel(r)
    held, entry = model.TABLES["live"], model.TABLES["live"]["k"][str(k)]
    tab_entry, level = tables["k"][str(k)], int(np.log2(m))
    pairs = np.array(held["pairs_by_bin"], float)
    weights = np.r_[pairs[:-1] / pairs[:-1].sum(), 0]
    a = model.floor_rates(entry, held, 0, 2, weights)
    b = model.floor_rates(tab_entry, tables, level, 2, weights)
    errors["recall"] += list(a[0] - b[0])
    common = (a[2] >= COMMON) & (a[0] > 0.05)
    errors["others"] += list(np.log(b[2][common] / a[2][common]))
    print(f"k{k} m{m}: recall {a[0][0]:.3f} / {b[0][0]:.3f}, others {a[2][0]:.2f} / {b[2][0]:.2f}, "
          f"postings {entry['postings']:,} / {tab_entry['postings']:,}")  # fmt: skip
summary = {key: float(np.sqrt(np.mean(np.square(v)))) for key, v in errors.items()}
print(f"against tables: recall RMSE {summary['recall']:.3f}, others log-ratio RMSE {summary['others']:.3f}")
assert summary["recall"] < 0.04 and summary["others"] < 0.3

# Parameters the tables do not cover respond as they should.
base = panel(k=9, m=8)
assert base["available"] and 0 < base["precision"] < 1 and 0 < base["auprc"] <= 1
high = panel(k=9, m=8, floor=7)
assert high["target_recall"] < base["target_recall"] and high["precision"] > base["precision"]
best = panel(k=9, m=8, margin=3)
assert best["precision"] > base["precision"] and best["recall"] <= base["recall"]
assert panel(k=9, m=8, no_relative=0.9)["precision"] < base["precision"]
assert panel(k=9, m=8192)["available"]
assert list(np.diff(base["target_by_floor"], axis=0).min(1) >= -1e-9) == [True] * 6  # more coverage, no less recall
assert panel(k=10, m=8)["available"]  # a k with no table
assert panel(k=9, m=8, length=100)["recall"] > base["recall"]
assert panel(k=9, m=8, paired=False)["recall"] < base["recall"]
assert panel(k=9, m=32)["recall"] < base["recall"]
reduced = panel(k=11, m=8, alphabet="murphy10")
assert 0 < reduced["recall"] <= 1 and reduced["chance"] >= 0

# Recall per target against direct simulation: random reads at a coverage, the same salts and floors.
errors = []
for k, m, coverage, f in [(7, 8, 1, 0), (9, 8, 2, 0), (9, 32, 5, 0), (9, 8, 1, 5), (9, 32, 10, 5), (11, 32, 20, 3), (7, 64, 50, 5), (11, 64, 100, 5)]:
    r = request(k=k, m=m, coverage=coverage, floor=f, mix="uniform", no_relative=0)
    got = json.loads(live.panel(r) if live.have(r) else (live.compute(r), live.panel(r))[1])
    d = live.DATA["mini-slice"]
    keep = np.flatnonzero(d["best"] >= 0)
    keep = keep[np.linspace(0, len(keep) - 1, 300).astype(int)]  # enough queries, at a coverage's reads
    rng = np.random.default_rng(k + m + coverage)
    index = live.build_index("mini-slice", "protein", k)
    keys, bits = index[0], index[4]
    charge = float(np.log2(2 * live.TRANSLATED_FRAMES * (50 - k + 1)))
    level_at = int(np.log2(m))
    called = []
    for salt in range(1, 9):
        read_q, reads = [], []
        for q in keep:
            s = d["query_list"][q]
            n = rng.poisson(coverage * len(s) / 100)
            for _ in range(n):
                gap = int(rng.integers(0, min(40, len(s) - 100) + 1))
                start = int(rng.integers(0, len(s) - 100 - gap + 1))
                read_q.append(q)
                reads += [s[start : start + 50], s[start + 50 + gap : start + 100 + gap]]
        read_q = np.array(read_q, np.int32)
        read, key = live.true_group_hits("mini-slice", "protein", k, reads, read_q, True)
        lv = live.level(keys ^ (salt * 0x51ED27))
        score, _, _ = live.scored(read, key, bits, lv, level_at, len(read_q), charge)
        passed = np.unique(read_q[score >= got["floors"][f]])
        for b in range(9):
            members = keep[d["bin"][keep] == b]
            if len(members):
                called.append((b, np.isin(members, passed).mean()))
    by_bin = np.array([np.mean([x for b, x in called if b == i]) for i in sorted({b for b, _ in called})])
    predicted = np.array(got["target_by_identity"][got["coverages"].index(coverage)])[sorted({b for b, _ in called})]
    errors += list(by_bin - predicted)
    print(f"k{k} m{m} {coverage}x floor {got['floors'][f]}: simulated {by_bin.mean():.3f}, table {predicted.mean():.3f}")
rmse = float(np.sqrt(np.mean(np.square(errors))))
print(f"recall per target against simulation: RMSE {rmse:.3f} over bins")
assert rmse < 0.1

# Long k-mers (hashed above k 14) and short ones (capped read pairs).
by_k = {k: panel(k=k, m=1, floor=0) for k in (31, 20, 15, 14, 4, 3, 2, 1)}
for k, q in by_k.items():
    print(f"k{k}: recall per target {q['target_recall']:.3f}, per read {q['recall']:.3f}, precision {q['precision']:.4f}, "
          f"{q['measured_pairs']} read pairs measured")
assert by_k[31]["target_recall"] < by_k[20]["target_recall"] < by_k[14]["target_recall"]
assert abs(by_k[15]["target_recall"] - by_k[14]["target_recall"]) < 0.03
assert by_k[4]["measured_pairs"] < 4500 and by_k[1]["recall"] > 0.8  # any hit counts with no floor
assert panel(k=5, m=1, floor=0, alphabet="dayhoff")["measured_pairs"] < 4500

# Combination k-mers on demand, against the native simulation (other queries).
grid = json.loads((ROOT / "checks/data/sim-combo.json").read_text())
errors, ratios = [], []
for row in grid["rows"]:
    if row["n"] < 2:
        continue
    if row["feature_sets"] > 40:
        continue
    got = json.loads(live.combo_simulate(request(k=row["k"], combo_n=row["n"], combo_lo=row["lo"], combo_hi=row["hi"])))
    gap = np.array(got["recall"])[:8] - np.array(row["recall"])
    errors.append(gap.ravel())
    if got["matches"] >= 50 and row["chance_matches"] >= 50:
        ratios.append(got["chance_m1"] / row["chance_per_pair_m1"])
    print(f"k{row['k']} n{row['n']} {row['lo']}-{row['hi']}: {got['pairs']} pairs, recall gap {np.sqrt(np.mean(gap**2)):.3f}, "
          f"chance {got['chance_m1']:.3g} / {row['chance_per_pair_m1']:.3g} from {got['matches']}, index share {got['index_share']:.2f}")
rmse = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
print(f"on demand against native: recall RMSE {rmse:.3f}; chance ratio {min(ratios):.2f} to {max(ratios):.2f}")
assert rmse < 0.06 and 0.5 < min(ratios) and max(ratios) < 2
wide = json.loads(live.combo_simulate(request(k=4, combo_n=3, combo_lo=1, combo_hi=60)))
print(f"wide window: {wide['sets']} sets, {wide['pairs']} pairs, index share {wide['index_share']:.3f}")
assert wide["pairs"] < 4500 and wide["index_share"] < 1

# A stored result gives the same panel, without the sequences.
r = request(k=9, m=16)
text = live.compute(r)
first = live.panel(r)
live.ENTRIES.clear()
live.add(r, text)
assert live.have(r) and live.panel(r) == first
print("ok")
