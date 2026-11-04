"""The simulation grid against the model and kfp (docs/plan.md, Progress log, phase 3).

Run: uv run --no-project --with numpy python checks/test_grid.py"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "web"))
import model  # noqa: E402

grid = json.loads((ROOT / "web/data/sim-grid.json").read_text())
ROWS = {(r["k"], r["m"]): r for r in grid["rows"]}
IDENTITY = np.array(grid["mean_identity"])
RELATIVES = np.array(grid["relative_bins"][:-1])
X_50 = int(np.searchsorted(RELATIVES, 0.5))
BIN_90 = 6  # identity 0.9-0.95


def rmse_rows(rows, ids, within):
    errors = [
        model.recall(50, r["k"], r["m"], ids, True, model.CONSERVATION, 1.0, model.READ_RATE, within) - np.array(r["recall"])
        for r in rows
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def main():
    errors = {"conserved": [], "independent": []}
    for (k, m), r in ROWS.items():
        for name, c in (("conserved", model.CONSERVATION), ("independent", model.INDEPENDENT)):
            shape = model.READ_RATE if name == "conserved" else None
            errors[name].append(model.recall(50, k, m, IDENTITY, True, c, 1.0, shape) - np.array(r["recall"]))
    conserved = float(np.sqrt(np.mean(np.square(errors["conserved"]))))
    independent = float(np.sqrt(np.mean(np.square(errors["independent"]))))
    bias = np.mean(errors["conserved"], axis=0)
    print(f"recall rmse: conserved {conserved:.3f}, independent {independent:.3f}")
    print("model minus simulation by identity bin", bias.round(3).tolist())
    assert conserved < 0.06 and independent > 0.1
    assert (np.abs(bias[4:]) < 0.03).all()  # identity >= 0.8
    assert bias[1:5].max() < -0.02  # known: the model is too low at identity 0.6-0.85

    # kfp's scaling table at k 9, on a similar number of groups (45,502 against 38,270).
    r = ROWS[9, 8]
    chance = r["decoy_anchors_per_pair"] * 3  # kfp counts 3 frames
    print(f"k9 m8 chance anchors, 3 frames: {chance:.4f} (kfp 0.014)")
    assert abs(chance / 0.014 - 1) < 0.5

    # Precision follows once the hits outside the aligned relatives are counted.
    x_mid = np.array([0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    weights = np.array(grid["relatives_per_query"])[:, 1:8]
    for k, m in [(6, 8), (7, 8), (9, 8), (9, 32), (11, 8)]:
        r = ROWS[k, m]
        other = np.array(r["other_per_pair"])
        seen = np.array(r["recall"])
        simulated = np.mean(seen / (seen + other.sum(1)))
        true = model.recall(50, k, m, IDENTITY, True, model.CONSERVATION, 1.0)
        relatives = (weights * model.recall(50, k, m, x_mid, True, model.CONSERVATION, 1.0)).sum(1)
        chance = 2 * (50 - k + 1) * r["postings"] * model.S2**k * model.HOT.get(k, 1)
        predicted = np.mean(true / (true + relatives + other[:, -1] + chance))
        print(f"k{k} m{m} precision: simulated {simulated:.3f}, predicted {predicted:.3f}, "
              f"hits outside the aligned relatives {np.mean(other[:, -1] / other.sum(1)):.2f}")
        assert abs(predicted / simulated - 1) < 0.2
    for k, m, seen in [(9, 8, 0.97), (9, 16, 0.88)]:
        got = ROWS[k, m]["recall"][BIN_90]
        print(f"k{k} m{m} pair recall at 0.9: {got:.2f} (kfp {seen})")
        assert 0 <= got - seen < 0.08  # no verification and no group distance here

    # The measured corrections, from an independent sample.
    for k, m in [(6, 8), (7, 8), (8, 8), (9, 8)]:
        r = ROWS[k, m]
        ratio = r["decoy_anchors_per_pair"] / (2 * (50 - k + 1) * r["postings"] * model.S2**k)
        print(f"k{k} m{m} chance / composition: {ratio:.2f} (kfp {model.HOT[k]})")
        assert 0.8 < ratio / model.HOT[k] < 1.6

    # Reduced alphabets, simulated on the same relatives. WITHIN_SCALE came from murphy10 and dayhoff.
    for name, limit in (("murphy10", 0.06), ("dayhoff", 0.07), ("seb8", 0.07)):
        reduced = json.loads((ROOT / f"web/data/sim-grid-{name}.json").read_text())
        ids = np.array(reduced["mean_identity"])
        letters, s2, within = model.alphabet_stats(name)
        scaled = rmse_rows(reduced["rows"], ids, within)
        unscaled = rmse_rows(reduced["rows"], ids, within / model.WITHIN_SCALE)
        print(f"{name}: within {within:.3f}, recall rmse {scaled:.3f}; BLOSUM62 alone {unscaled:.3f}")
        assert scaled < limit and scaled < unscaled
        for r in reduced["rows"]:
            ratio = r["decoy_anchors_per_pair"] / (2 * (50 - r["k"] + 1) * r["postings"] * s2 ** r["k"])
            if r["m"] == 8 and r["k"] <= 11:
                print(f"  k{r['k']} m8 chance / class composition: {ratio:.2f}")
                assert 0.7 < ratio < 2.5

    # Hits on relatives are not independent of the hit on the true group.
    for k, m in [(7, 8), (9, 8), (9, 32), (11, 8)]:
        r = ROWS[k, m]
        hit = sum(r["other_given_anchor"][X_50:-1]) / sum(r["relatives_given_anchor"][X_50:-1])
        miss = sum(r["other_given_miss"][X_50:-1]) / sum(r["relatives_given_miss"][X_50:-1])
        print(f"k{k} m{m} relative hit at x >= 0.5: {hit:.3f} if true group anchored, {miss:.3f} if not")
        assert hit > 1.5 * miss
    print("ok")


if __name__ == "__main__":
    main()
