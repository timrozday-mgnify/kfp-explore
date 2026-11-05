# kfp-explore

An interactive web page for exploring k-mer methods: how k, sparsity and the noise floor change
precision and recall, for different index sizes, read lengths and homologue structures. It is based
on the methods of [kmer-functional-profiler](https://github.com/timrozday-mgnify/kmer-functional-profiler).

Early stage: see [the plan](docs/plan.md). The page shows anchor recall, index size and chance
anchors from an analytic model, and analyses a small protein set (a bundled sample or your own
FASTA). Both run in the browser with NumPy in Pyodide.

```bash
python3 -m http.server 8765 --bind 127.0.0.1 -d web  # then open http://127.0.0.1:8765/
```

```bash
uv run --no-project --with numpy python checks/test_model.py  # model against Monte Carlo and kfp's tables
uv run --no-project --with numpy python checks/test_analyse.py  # analysis against the pilot and random sequences
uv run --no-project --with numpy python checks/test_grid.py  # simulation grid against the model and kfp
uv run --no-project --with numpy python checks/test_tables.py  # precision tables against held-out simulation
uv run --no-project --with numpy python checks/test_benchmark.py  # kfp benchmark table against kfp's gate
uv run --no-project --with numpy python checks/test_policy.py  # grouping and key policy against kfp's comparisons
uv run --no-project --with numpy python checks/test_live.py  # on-demand precision against the tables
uv run --no-project --with numpy python checks/test_snp.py  # SNP panel against brute force and simulation
uv run --no-project --with numpy python checks/test_combo.py  # combination k-mers against simulation and kfp
uv run --no-project --with numpy python measure/fit_rate.py web/data/sim-grid.json web/data/sim-grid-slice.json  # fit the read rate on each set
```

Licence: GPL-3.0-or-later.
