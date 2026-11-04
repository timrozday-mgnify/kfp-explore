"""The model against Monte Carlo and kfp's measured tables (docs/plan.md).

Run: uv run --no-project --with numpy python checks/test_model.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "web"))
import model  # noqa: E402

# kfp's 90%-AAI table: k, m, release postings, chance anchors per pair, pair recall at P.
TABLE = [
    (6, 8, 3e10, 4.6e5, [0.96, 0.96, 0.97, 0.93]),
    (7, 8, 3e10, 3.3e4, [0.95, 0.95, 0.96, 0.90]),
    (8, 8, 3e10, 1870, [0.95, 0.96, 0.94, 0.87]),
    (9, 8, 3e10, 274, [0.94, 0.95, 0.93, 0.82]),
    (8, 16, 1.5e10, 773, [0.91, 0.90, 0.84, 0.72]),
    (9, 16, 1.5e10, 129, [0.88, 0.86, 0.80, 0.65]),
]
P = np.array([0.95, 0.9, 0.85, 0.8])


def monte_carlo(length, k, m, p, conservation, n=200_000, seed=1):
    """P(no anchor), simulated with the chain's own assumptions."""
    rng = np.random.default_rng(seed)
    phi, rho, ell = conservation
    id0, id1 = model.classes(p, phi, rho)
    to_v, to_c = 1 / ell, 1 / ell * phi / (1 - phi)
    conserved = rng.random(n) < phi
    run, hit = np.zeros(n, int), np.zeros(n, bool)
    for _ in range(length):
        conserved ^= rng.random(n) < np.where(conserved, to_v, to_c)
        same = rng.random(n) < np.where(conserved, id0, id1)
        run = (run + 1) * same
        hit |= (run >= k) & (rng.random(n) < 1 / m)
    return 1 - hit.mean()


def rmse(rows, conservation, ceiling):
    errors = [
        model.recall(50, k, m, P, True, conservation, ceiling) - np.array(seen)
        for k, m, _, _, seen in rows
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def fit(rows):
    grid = [
        ((phi, rho, ell), ceiling)
        for phi in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        for rho in (2, 4, 8, 16, 32, 64)
        for ell in (5, 10, 20, 40, 80)
        for ceiling in (0.95, 0.96, 0.97, 0.98, 1)
    ]
    return min(grid, key=lambda g: rmse(rows, *g))


def main():
    for length, k, m, p, conservation in [
        (50, 9, 8, 0.9, model.INDEPENDENT),
        (50, 9, 16, 0.8, model.CONSERVATION),
        (100, 5, 8, 0.7, model.CONSERVATION),
        (30, 11, 4, 0.95, (0.8, 32.0, 10.0)),
    ]:
        chain = float(model.miss(length, k, m, p, conservation))
        simulated = monte_carlo(length, k, m, p, conservation)
        print(f"chain {chain:.4f}, simulated {simulated:.4f}: L{length} k{k} m{m} p{p}")
        assert abs(chain - simulated) < 0.005

    assert abs(model.miss(5, 9, 1, 1.0) - 1) < 1e-9  # read shorter than k
    assert abs(model.miss(50, 9, 1, 1.0)) < 1e-9  # identical read, nothing sampled away

    train = [r for r in TABLE if r[1] == 8]
    test = [r for r in TABLE if r[1] == 16]
    conservation, ceiling = fit(train)
    held_out = rmse(test, conservation, ceiling)
    independent = rmse(TABLE, model.INDEPENDENT, 1)
    print(f"fit {conservation}, ceiling {ceiling}: train {rmse(train, conservation, ceiling):.3f}, "
          f"held-out m 16 {held_out:.3f}, independent {independent:.3f}")
    assert (conservation, ceiling) == (model.CONSERVATION, model.CEILING)
    assert held_out < 0.04 and independent > 0.06

    for k, m, n_postings, seen, _ in TABLE:
        base, corrected = model.chance_anchors(50, k, n_postings)
        print(f"k{k} m{m} chance: composition {base:.3g}, corrected {corrected:.3g}, kfp {seen:.3g}")
        assert 1 < seen / base < 5
        assert 0.5 < seen / corrected < 2

    for k, m, n_postings, _, _ in TABLE[3:5]:
        got = model.postings(5.74e9, 0.155, m)
        print(f"k{k} m{m} postings: model {got:.3g}, kfp {n_postings:.3g}")
        assert 0.7 < got / n_postings < 1.3

    print("ok")


if __name__ == "__main__":
    main()
