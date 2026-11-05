#!/usr/bin/env bash
#SBATCH --job-name=kfp-scale-profile
#SBATCH --output=%x.%j.log
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=ALL
# Scale profile of MGnify (kfp-explore, docs/plan.md, Scaling to full size). Clone kfp-explore beside
# kmer-functional-profiler, then:
#   mkdir -p <runs>/kfp-ablations/scale-profile && cd <runs>/kfp-ablations/scale-profile
#   cp ../../../kfp-explore/measure/scale-profile/run.sh . && sbatch run.sh
# Reads release-tiers' 256 members buckets. Re-submitting resumes.
# Bring back: results/scale-profile.json, results/check.tsv, results/trace.tsv
set -euo pipefail
KFP=../../../kmer-functional-profiler
EXPLORE=../../../kfp-explore
[[ -f run.sh && -f ../site.config ]] || { echo "submit from the run's directory" >&2; exit 1; }
[[ -x $KFP/.venv/bin/python ]] || { echo "no venv in $KFP: run its workflows/setup.sh first" >&2; exit 1; }
STORE=$(realpath ../release-tiers/results-standard)
compgen -G "$STORE/members.bucket*.parquet" >/dev/null || { echo "no $STORE/members.bucket*.parquet: run release-tiers first" >&2; exit 1; }
export NXF_OPTS='-Xms1g -Xmx4g'
nextflow run "$EXPLORE/measure/scale-profile/main.nf" -c ../site.config -resume --outdir results \
    --store "$STORE/members.bucket*.parquet" --venv "$(realpath "$KFP/.venv")"
