"""Fit the read rate (and optionally conservation) to simulation grids, and test what it predicts.

The shape is fitted to true-group recall at m 8, then used to predict the rest of the grid and hits
on relatives. Run: uv run --no-project --with numpy python measure/fit_rate.py web/data/sim-grid*.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "web"))
import model  # noqa: E402

SHAPES = (1, 1.5, 2, 3, 4, 6, 8, 12, 20, 50)
X_MID = np.array([0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
SHARES = (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
TEST_ROWS = [(7, 8), (9, 8), (9, 32), (11, 8)]


def load(path):
    grid = json.loads(Path(path).read_text())
    grid["by"] = {(r["k"], r["m"]): r for r in grid["rows"]}
    return grid


def rmse(grid, rows, shape, conservation=model.CONSERVATION):
    ids = np.array(grid["mean_identity"])
    errors = [
        model.recall(50, k, m, ids, True, conservation, 1.0, shape) - np.array(grid["by"][k, m]["recall"])
        for k, m in rows
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def split(grid):
    train = [x for x in grid["by"] if x[1] == 8 and x[0] in (6, 7, 8, 9)]
    return train, [x for x in grid["by"] if x not in train]


def fit_shape(grid, conservation=model.CONSERVATION):
    return min(SHAPES, key=lambda a: rmse(grid, split(grid)[0], a, conservation))


def aligned_hits(grid, k, m, shape):
    """Aligned relatives hit per pair, simulated and predicted (bins from 0.3 up)."""
    weights = np.array(grid["relatives_per_query"])[:, 1:8]
    hit = model.recall(50, k, m, X_MID[1:], True, model.CONSERVATION, 1.0, shape)
    seen = np.array(grid["by"][k, m]["other_per_pair"])[:, 1:8].sum(1).mean()
    return float(seen), float((weights * hit).sum(1).mean())


def after_miss(grid, k, m, shape, share):
    """P(a relative is hit | true group missed) per relative bin from 0.4, predicted and simulated."""
    ids = np.array(grid["mean_identity"])
    weights = np.array(grid["relatives_per_query"])[:, :8]
    r = grid["by"][k, m]
    seen = np.array(r["other_given_miss"])[:8] / np.maximum(np.array(r["relatives_given_miss"])[:8], 1)
    p, x = np.meshgrid(ids, X_MID[2:], indexing="ij")
    missed, both = model.hit_both(50, k, m, p, x, np.broadcast_to(share, p.shape), shape=shape)
    w = weights[:, 2:]
    return (w * both).sum(0) / np.maximum((w * missed).sum(0), 1e-12), seen[2:]


def fit_shares(grid, shape, k=9, m=8):
    """The share per relative bin at one k and m, for the other rows and grids to test."""
    candidates = {s: after_miss(grid, k, m, shape, s) for s in SHARES}
    return np.array([
        min(SHARES, key=lambda s: abs(candidates[s][0][i] - candidates[s][1][i]))
        for i in range(len(X_MID) - 2)
    ])


def main():
    shares = fit_shares(load(sys.argv[1]), model.READ_RATE)
    print(f"shares by relative bin, from {sys.argv[1]} at k9 m8: {shares.tolist()}")
    fits = {}
    for path in sys.argv[1:]:
        grid = load(path)
        name = Path(path).stem
        train, rest = split(grid)
        shape = fit_shape(grid)
        fits[name] = shape
        print(f"\n{name}: {grid['groups']:,} groups, {grid['queries']} queries, top {grid['top']}")
        print(f" shape {shape}; recall rmse train {rmse(grid, train, shape):.3f}, rest {rmse(grid, rest, shape):.3f}; "
              f"no rate: train {rmse(grid, train, None):.3f}, rest {rmse(grid, rest, None):.3f}")
        for k, m in TEST_ROWS:
            seen, got = aligned_hits(grid, k, m, shape)
            print(f" k{k} m{m} aligned relatives hit per pair: simulated {seen:.2f}, model {got:.2f}")
        for k, m in TEST_ROWS:
            got, seen = after_miss(grid, k, m, shape, shares)
            free, _ = after_miss(grid, k, m, shape, 0)
            print(f" k{k} m{m} hit after a miss, x 0.4+: simulated {seen.round(3)}")
            print(f"           shared hash {got.round(3)}; without it {free.round(3)}")
            print(f"           rmse {np.sqrt(np.mean((got - seen) ** 2)):.3f}; without {np.sqrt(np.mean((free - seen) ** 2)):.3f}")
    grids = [load(p) for p in sys.argv[1:]]
    if len(grids) > 1:
        print("\nTransfer: each grid's recall under the other's shape")
        names = [Path(p).stem for p in sys.argv[1:]]
        for name, grid in zip(names, grids, strict=True):
            for other in names:
                _, rest = split(grid)
                print(f" {name} with {other}'s shape {fits[other]}: rest of grid rmse {rmse(grid, rest, fits[other]):.3f}")


if __name__ == "__main__":
    main()
