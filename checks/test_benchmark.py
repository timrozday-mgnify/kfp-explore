"""The benchmark table against the gate kfp computed from the full rankings, and its size fit."""

import json
from pathlib import Path

t = json.loads((Path(__file__).parent.parent / "web/data/kfp-benchmark.json").read_text())
step = t["precision_step"]
worst = 0
for row in t["gate"]:
    grid = t["runs"][row["tree"]]["at_precision"][row["level"]]
    i = int(row["precision"] / step - 1e-9) + 1  # the next grid value up
    below = grid[i - 1]  # recall can only fall with precision, so the two bracket the gate's value
    above = grid[i] if i < len(grid) else 0.0
    assert above - 1e-3 <= row["value"] <= below + 1e-3, row
    worst = max(worst, below - above)
fit = t["size_fit"]
assert 15 < fit["bytes_per_posting"] < 25
for name, tier in t["tiers"].items():
    predicted = fit["fixed_gb"] + tier["postings"] * fit["bytes_per_posting"] / 1e9
    assert abs(predicted / (tier["bytes"] / 1e9) - 1) < 0.06, name
print(f"{len(t['gate'])} gate rows reproduced; widest bracket {worst:.3f} recall")
