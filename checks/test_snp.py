"""The SNP and overlap panel (docs/plan.md, phase 7) against brute force and simulation.

Run: uv run --no-project --with numpy python checks/test_snp.py"""

import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "web"))
import model  # noqa: E402

rng = np.random.default_rng(1)

# Fewest positions and the removal curve, against every set of positions on small reads.
for _ in range(200):
    n = int(rng.integers(5, 16))
    k = int(rng.integers(1, 5))
    starts = np.flatnonzero(rng.random(n - k + 1) < rng.choice([0.15, 0.4, 0.8]))
    first, last = starts, starts + k - 1
    hit_all = lambda chosen: all(any(a <= p <= b for p in chosen) for a, b in zip(first, last, strict=True))  # noqa: E731
    fewest = next(d for d in range(n + 1) if any(hit_all(c) for c in itertools.combinations(range(n), d)))
    hits = model.fewest_to_remove(first, last)
    assert len(hits) == fewest and hit_all(hits), (first, last, hits, fewest)
    curve = model.removal_curve(first, last, n, n)
    for d in range(n + 1):
        exact = sum(hit_all(c) for c in itertools.combinations(range(n), d)) / comb(n, d)
        assert abs(curve[d] - exact) < 1e-9, (first, last, d, curve[d], exact)
    assert (np.diff(curve) >= -1e-9).all()

# A read's chance of holding no key, against simulated reads.
for placement, length, k, m, paired in [
    ("sampled", 50, 9, 8, False), ("sampled", 50, 9, 32, True), ("even", 50, 9, 64, False), ("even", 50, 9, 16, False),
    ("minimizer", 30, 9, 32, False), ("minimizer", 50, 9, 16, False), ("minimizer", 20, 9, 16, True), ("minimizer", 50, 9, 64, False),
]:  # fmt: skip
    draws = 4000
    empty = np.mean([
        len(model.read_keys(length, k, m, placement, paired, np.random.default_rng(i))[0]) == 0 for i in range(draws)
    ])  # fmt: skip
    predicted = model.no_key_chance(length, k, m, placement, paired)
    print(f"{placement} L{length} k{k} m{m} paired {paired}: no key {empty:.3f} simulated, {predicted:.3f} predicted")
    assert abs(empty - predicted) < 0.03, placement

# SNP to residue.
assert abs(model.residue_rate(0.01, 0, "protein", 0) - (1 - 0.99**3) * 0.75) < 1e-12
assert abs(model.residue_rate(0.01, 0, "dna", 0) - 0.01) < 1e-12 and model.residue_rate(0.01, 0, "protein", 0.5) < 0.0223 / 2 + 1e-3

# The panel as a whole, and uniform substitutions against simulation.
a = {"length": 50, "k": 9, "m": 8, "paired": True, "placement": "sampled", "draw": 1, "d": 3, "snp": 0.03, "error": 0.0, "alphabet": "protein"}  # fmt: skip
q = model.snp_panel(a, 0.0)
assert q["mean"][0] == min(q["mean"]) and 0 <= q["recall_uniform"] <= 1 and 0 <= q["recall_chain"] <= 1
survived = []
sim = np.random.default_rng(5)
for i in range(3000):
    first, last = model.read_keys(50, 9, 8, "sampled", True, np.random.default_rng(10_000 + i % model.MEAN_DRAWS))
    changed = np.flatnonzero(sim.random(100) < q["rate"])
    survived.append(any(not ((a0 <= changed) & (changed <= b0)).any() for a0, b0 in zip(first, last, strict=True)))
print(f"recall under uniform substitutions: {np.mean(survived):.3f} simulated, {q['recall_uniform']:.3f} computed; chain {q['recall_chain']:.3f}")
assert abs(np.mean(survived) - q["recall_uniform"]) < 0.02
print("ok")
