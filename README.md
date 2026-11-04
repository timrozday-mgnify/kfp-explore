# kfp-explore

An interactive web page for exploring k-mer methods: how k, sparsity and the noise floor change
precision and recall, for different index sizes, read lengths and homologue structures. It is based
on the methods of [kmer-functional-profiler](https://github.com/timrozday-mgnify/kmer-functional-profiler).

Early stage: see [the plan](docs/plan.md). The page does not exist yet.

```bash
node checks/model_check.mjs  # the model against Monte Carlo and kfp's measured tables
```
