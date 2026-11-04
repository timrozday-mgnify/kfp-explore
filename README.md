# kfp-explore

An interactive web page for exploring k-mer methods: how k, sparsity and the noise floor change
precision and recall, for different index sizes, read lengths and homologue structures. It is based
on the methods of [kmer-functional-profiler](https://github.com/timrozday-mgnify/kmer-functional-profiler).

Early stage: see [the plan](docs/plan.md). The page shows anchor recall, index size and chance
anchors from an analytic model, which runs in the browser with NumPy in Pyodide.

```bash
python3 -m http.server 8765 --bind 127.0.0.1 -d web  # then open http://127.0.0.1:8765/
```

```bash
uv run --no-project --with numpy python checks/test_model.py  # model against Monte Carlo and kfp's tables
```

Licence: GPL-3.0-or-later.
