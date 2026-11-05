"""Precision tables against held-out simulation and kfp's own measurements (docs/plan.md, phase 4).

Tables come from one set of queries; the held-out run uses different queries. Both are means over hash
salts. kfp's numbers are from docs/tree-index-plan.md and the HPC runs' indexes.

Run: uv run --no-project --with numpy python checks/test_tables.py"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "web"))
import model  # noqa: E402

COMMON = 1.0  # others per pair above which a setting is compared closely
FLOORS = [0, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200]
LEVELS = (0, 3, 5, 7)


def rates(entry, t, level, truth, weights, rule=None):
    true_hits, ok, wrong = model.floor_rates(entry, t, level, truth, weights, rule)
    return true_hits, ok, wrong


def held_out(t, held):
    pairs = np.array(held["pairs_by_bin"], float)
    all_reads = pairs / pairs.sum()
    relatives = np.r_[pairs[:-1] / pairs[:-1].sum(), 0]
    for rule, name in ((None, "every group over the floor"), (1, "best group within 3 bits")):
        errors = {"recall": [], "others": [], "precision": [], "precision_70": []}
        print(f"\nHeld-out, {name}: floor: recall held / table; others; precision; precision at x >= 0.7")
        for k in sorted(held["k"], key=int):
            a, b = t["k"][k], held["k"][k]
            for level in LEVELS:
                for f, floor in enumerate(FLOORS):
                    seen, tab = [rates(e, x, level, 2, relatives, rule) for e, x in ((b, held), (a, t))]
                    recall, others = (seen[0][f], tab[0][f]), (seen[2][f], tab[2][f])
                    p = [rates(e, x, level, 2, all_reads, rule) for e, x in ((b, held), (a, t))]
                    precision = [q[0][f] / max(q[0][f] + q[2][f], 1e-12) for q in p]
                    p70 = [rates(e, x, level, 0.7, all_reads, rule) for e, x in ((b, held), (a, t))]
                    lenient = [(q[0][f] + q[1][f]) / max(q[0][f] + q[1][f] + q[2][f], 1e-12) for q in p70]
                    errors["recall"].append(tab[0][f] - seen[0][f])
                    if seen[2][f] >= (COMMON if rule is None else 0.2) and seen[0][f] > 0.05:
                        errors["others"].append(np.log(others[1] / others[0]))
                        errors["precision"].append(np.log(precision[1] / precision[0]))
                        errors["precision_70"].append(np.log(lenient[1] / lenient[0]))
                    if level == 3 and floor in (0, 20, 40, 80, 120):
                        print(f"  k{k} m8 {floor:3d}: {recall[0]:.3f} / {recall[1]:.3f}; {others[0]:6.2f} / {others[1]:6.2f}; "
                              f"{precision[0]:.3f} / {precision[1]:.3f}; {lenient[0]:.3f} / {lenient[1]:.3f}")
        summary = {key: float(np.sqrt(np.mean(np.square(v)))) for key, v in errors.items()}
        print(f" rmse: recall {summary['recall']:.4f}; log ratio, {len(errors['others'])} common settings:",
              {key: round(summary[key], 3) for key in ("others", "precision", "precision_70")})
        assert summary["recall"] < 0.02
        assert summary["others"] < 0.2 and summary["precision"] < 0.25 and summary["precision_70"] < 0.1


def per_target(t, held):
    depth = held["pairs_per_query"]
    queries = np.array(held["queries_by_bin"], float)
    weights = queries / queries.sum()
    errors = []
    print(f"\nPer target at depth {depth}, no floor: true called, one hash draw / table; false groups")
    for k in sorted(held["k"], key=int):
        entry = t["k"][k]
        for level, h in [(3, 1), (5, 1)]:
            r = held["k"][k]["real"][f"{level},{h}"]
            called, _, wrong = model.target_rates(entry, t, level, 0, 2, weights, depth)
            print(f"  k{k} m{2**level}: {r['target_true']:.3f} / {called:.3f}; {r['target_false']:.1f} / {wrong:.1f}")
            errors.append(called - r["target_true"])
    print(f" rmse called {np.sqrt(np.mean(np.square(errors))):.3f}")
    assert np.sqrt(np.mean(np.square(errors))) < 0.08


def against_kfp(t, slice_t):
    print("\nAgainst kfp's measurements, nothing fitted")
    # Score per hit at k 5: kfp's fit is 9 + 8 per hit (wide k 5, 1 key per ~12 residues).
    e = t["k"]["5"]
    bits = np.mean(e["bits_per_true_hit"][3:5])
    print(f" k5 score: {e['charge']:.1f} + {bits - e['charge']:.1f} per hit; kfp 9 + 8")
    assert abs(e["charge"] - 8.9) < 0.5 and abs(bits - e["charge"] - 8) < 1.5
    # Mean key bits: kfp's slice index, k 7 and k 11 keys, against 1/16 of the same slice.
    for k, seen in (("7", 29.1), ("11", 45.4)):
        got = slice_t["k"][k]["key_bits"][0]
        print(f" k{k} mean key bits: {got:.1f}; kfp {seen}")
        assert abs(got - seen) < 1.5
    # Decoys at k 5: kfp's 261,591 read pairs give 23,364 / 1,947 leaves at 30 / 40 bits, on one
    # index with one key per ~12 residues (between m 8 and 16), counting all frames.
    for floor, seen in ((30, 23364 / 261591), (40, 1947 / 261591)):
        f = FLOORS.index(floor)
        low, high = [e["floor_decoys"][level, f] * model.FRAMES for level in (4, 3)]
        print(f" k5 decoys per pair at {floor} bits: {low:.4f} to {high:.4f} (m 16 to 8); kfp {seen:.4f}")
        assert low / 5 < seen < high * 5
    # True group over a high bar: kfp's confident reads at 90.5 bits, by identity.
    bins = [8, 7, 6, 4, 2]  # identity 0.98+, 0.95, 0.9, 0.8, 0.7
    seen = [0.77, 0.46, 0.29, 0.13, 0.02]
    pairs = np.array(t["pairs_by_bin"], float)
    for level in (3, 4):
        at_80, at_120 = [
            e["floor_cube"][level, FLOORS.index(floor), bins, 0] / pairs[bins] for floor in (80, 120)
        ]
        print(f" k5 m{2**level} true group at 120 to 80 bits: {at_120.round(2).tolist()} to {at_80.round(2).tolist()}; kfp at 90.5: {seen}")


def main():
    for name in ("mini-slice", "slice"):
        model.load_table(name, (ROOT / f"web/data/tables-{name}.json").read_text())
    model.load_table("held", (ROOT / "checks/data/heldout-mini-slice.json").read_text())
    t, held = model.TABLES["mini-slice"], model.TABLES["held"]

    # Rules on the same reads: best-group sets are subsets, tighter margins are smaller.
    for k, e in t["k"].items():
        all_rows = e["floor_cube"].sum((2, 3))
        by_margin = e["best_cube"].sum((3, 4))
        assert (by_margin[0] <= by_margin[1]).all() and (by_margin[1] <= by_margin[2]).all(), k
        assert (by_margin[2] <= all_rows).all(), k
        # A higher floor keeps fewer rows, and no floor keeps every hit.
        assert (np.diff(all_rows, axis=1) <= 0).all(), k

    held_out(t, held)
    per_target(t, held)
    against_kfp(t, model.TABLES["slice"])

    k9, low = t["k"]["9"]["floor_cube"][3], [0, 1, 2]
    assert (k9[low] == k9[0]).all()  # floors under a k 9 key's bits change nothing
    print("ok")


if __name__ == "__main__":
    main()
