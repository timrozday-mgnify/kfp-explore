"""Precision from shared k-mer tables against held-out simulation (docs/plan.md, phase 4).

Tables come from one set of queries; real hash sampling comes from different queries, averaged over
hash salts, as the tables give an expectation over hashes. Unsalted, every run is the same hash draw.

Run: uv run --no-project --with numpy python checks/test_tables.py"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "web"))
import model  # noqa: E402

HITS = (1, 2, 3, 5)
COMMON = 1.0  # others per pair above which a setting is compared closely


def main():
    for name in ("mini-slice", "slice"):
        model.load_table(name, (ROOT / f"web/data/tables-{name}.json").read_text())
    assert model.tail_table(3, 2, 2)[3, 0] == 1 - 0.5**3
    assert abs(model.tail_table(3, 2, 2)[3, 1] - 0.5) < 1e-12
    assert model.floor_hits(0, 9) == 1 and model.floor_hits(40, 9) == 2
    assert model.floor_hits(50, 1) == model.HIT_LIMIT + 1  # 1-mers carry too little to add up

    t = model.TABLES["mini-slice"]
    held = json.loads((ROOT / "checks/data/heldout-mini-slice.json").read_text())
    pairs = np.array(held["pairs_by_bin"], float)
    all_reads = pairs / pairs.sum()
    relatives = np.r_[pairs[:-1] / pairs[:-1].sum(), 0]
    errors = {"recall": [], "others": [], "precision": [], "precision_70": [], "target": []}
    for k in sorted(held["k"], key=int):
        entry = t["k"][k]
        real = held["k"][k]["real"]
        print(f"k{k}: level, h: recall held-out / table; others; precision; precision at x >= 0.7 (one draw)")
        for level in (0, 3, 5, 7):
            m = 2**level
            with_all = model.table_rates(entry, t, m, 5, 2, all_reads)
            lenient = model.table_rates(entry, t, m, 5, 0.7, all_reads)
            with_rel = model.table_rates(entry, t, m, 5, 2, relatives)
            for h in HITS:
                r = real[f"{level},{h}"]
                s_recall, s_others, s_true, s_all = held["k"][k]["salted"][f"{level},{h}"]
                s_precision = s_true / max(s_all, 1e-12)
                recall = with_rel[0][h - 1]
                others = with_rel[2][h - 1]
                tp, wrong = with_all[0][h - 1], with_all[2][h - 1]
                precision = tp / max(tp + wrong, 1e-12)
                ok = lenient[0][h - 1] + lenient[1][h - 1]
                p70 = ok / max(ok + lenient[2][h - 1], 1e-12)
                print(f"  m{m:<4d} h{h}: {s_recall:.3f} / {recall:.3f}; {s_others:.2f} / {others:.2f}; "
                      f"{s_precision:.3f} / {precision:.3f}; {r['precision_70']:.3f} / {p70:.3f}")
                errors["recall"].append(recall - s_recall)
                if s_others >= COMMON:
                    errors["others"].append(np.log(others / s_others))
                    errors["precision"].append(np.log(precision / s_precision))
                    errors["precision_70"].append(np.log(p70 / r["precision_70"]))
    summary = {key: float(np.sqrt(np.mean(np.square(v)))) for key, v in errors.items() if v}
    print("rmse: recall", round(summary["recall"], 4), "; log ratio, common settings:",
          {key: round(summary[key], 3) for key in ("others", "precision", "precision_70")})
    assert summary["recall"] < 0.02
    assert summary["others"] < 0.2 and summary["precision"] < 0.25 and summary["precision_70"] < 0.1

    # Per target, at the held-out run's depth, with every query counted once.
    depth = held["pairs_per_query"]
    queries = np.array(held["queries_by_bin"], float)
    weights = queries / queries.sum()
    target = []
    print(f"per target at depth {depth}: true called real / table; false groups")
    for k in sorted(held["k"], key=int):
        entry = t["k"][k]
        for level, h in [(3, 1), (3, 3), (5, 1), (5, 2)]:
            r = held["k"][k]["real"][f"{level},{h}"]
            called, ok, wrong = model.target_rates(entry, t, 2**level, h, 2, weights, depth)
            print(f"  k{k} m{2**level} h{h}: {r['target_true']:.3f} / {called:.3f}; {r['target_false']:.1f} / {wrong:.1f}")
            target.append((called - r["target_true"], np.log(max(wrong, 1e-9) / max(r["target_false"], 1e-9))))
    target = np.array(target)
    print(f"rmse: called {np.sqrt(np.mean(target[:, 0] ** 2)):.3f}, false groups log ratio {np.sqrt(np.mean(target[:, 1] ** 2)):.3f}")
    assert np.sqrt(np.mean(target[:, 0] ** 2)) < 0.08

    params = {
        "index": "mini-slice", "k": 9, "m": 8, "floor": 0, "truth": 0.7, "mix": "natural", "no_relative": 0.05,
        "reads": 1e7, "targets": 1e4,
    }
    q = model.compute_precision(params)
    assert q["available"] and 0 < q["precision"] < 1 and 0 < q["auprc"] <= 1 and q["hits_needed"] == 1
    assert not model.compute_precision(params | {"k": 10})["available"]
    high = model.compute_precision(params | {"floor": 100})
    assert high["hits_needed"] > 1 and high["recall"] < q["recall"] and high["precision"] > q["precision"]
    over = model.compute_precision(params | {"k": 5, "floor": 200})
    assert over["over_limit"] and over["recall"] == 0 and over["precision"] is None
    all_novel = model.compute_precision(params | {"no_relative": 0.9, "truth": 2})
    assert all_novel["precision"] < q["precision"]
    print("ok")


if __name__ == "__main__":
    main()
