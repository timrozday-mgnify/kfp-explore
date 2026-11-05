#!/usr/bin/env nextflow
// Scale profile of a protein release (docs/plan.md, Scaling to full size). See README.md.
// COUNT      per bucket: whole-release anchors
// SAMPLE     per bucket: the largest rung's proteins, by a hash of protein_id, and held-out queries
// UNITS      per rung: proteins, units, singletons, lengths, composition
// KMERS      per rung and k: distinct k-mers, holders, hot tail, decoys, matches between halves
// ALIGN      per rung: DIAMOND, the held-out queries against the rung
// SPECTRUM   per rung: other groups per query by identity
// SUMMARY    scale-profile.json for the page; check.tsv, the largest rung predicted from the others

nextflow.enable.dsl = 2

def py() { "${params.venv}/bin/python ${projectDir}/scale.py" }

process COUNT {
    tag "${bucket.name}"
    label 'process_medium'

    input:
    path bucket

    output:
    path 'count.json'

    script:
    "${py()} count --bucket ${bucket} --out count.json"
}

process TOTALS {
    label 'process_low'

    input:
    path counts, stageAs: 'count/?.json'

    output:
    path 'totals.json'

    script:
    """
    ${params.venv}/bin/python -c "
import json, glob
print(json.dumps({'proteins': sum(json.load(open(p))['proteins'] for p in glob.glob('count/*.json'))}))
" > totals.json
    """
}

process SAMPLE {
    tag "${bucket.name}"
    label 'process_medium'

    input:
    path bucket
    path totals

    output:
    path "sample-${bucket.baseName}.parquet", emit: rung
    path "queries-${bucket.baseName}.parquet", emit: queries

    script:
    """
    ${py()} sample --bucket ${bucket} --totals ${totals} --top ${params.rungs.tokenize(',')*.toDouble().max()} \
        --queries ${params.query_pool} --out sample-${bucket.baseName}.parquet --queries-out queries-${bucket.baseName}.parquet
    """
}

process UNITS {
    tag "${rung}"
    label 'process_medium'

    input:
    val rung
    path samples, stageAs: 'samples/*'
    path totals

    output:
    tuple val(rung), path("units-${rung}.json")

    script:
    "${py()} units --samples samples --totals ${totals} --rung ${rung} --out units-${rung}.json"
}

process KMERS {
    tag "${rung} k${k}"
    label 'process_high'

    input:
    tuple val(rung), path(units), val(k)
    path samples, stageAs: 'samples/*'
    path totals

    output:
    path "kmers-${rung}-${k}.json"

    script:
    "${py()} kmers --samples samples --totals ${totals} --rung ${rung} --k ${k} --units ${units} --out kmers-${rung}-${k}.json"
}

process QUERIES {
    label 'process_medium'

    input:
    path samples, stageAs: 'samples/*'

    output:
    path 'queries.fa'

    script:
    "${py()} queries --samples samples --count ${params.queries} --out queries.fa"
}

process FASTA {
    tag "${rung}"
    label 'process_medium'

    input:
    val rung
    path samples, stageAs: 'samples/*'
    path totals

    output:
    tuple val(rung), path("rung-${rung}.fa")

    script:
    "${py()} fasta --samples samples --totals ${totals} --rung ${rung} --out rung-${rung}.fa"
}

process ALIGN {
    tag "${rung}"
    label 'process_align'
    container 'quay.io/biocontainers/diamond:2.2.8--he361c42_0'

    input:
    tuple val(rung), path(fa)
    path queries

    output:
    tuple val(rung), path("hits-${rung}.tsv")

    script:
    """
    diamond makedb --in ${fa} --db rung --threads ${task.cpus}
    diamond blastp --db rung --query ${queries} --out hits-${rung}.tsv --threads ${task.cpus} \
        --outfmt 6 qseqid sseqid pident qcovhsp ${params.diamond_args}
    rm -f rung.dmnd
    """
}

process SPECTRUM {
    tag "${rung}"
    label 'process_medium'

    input:
    tuple val(rung), path(hits)
    path samples, stageAs: 'samples/*'
    path totals
    path queries

    output:
    path "spectrum-${rung}.json"

    script:
    """
    ${py()} spectrum --samples samples --totals ${totals} --rung ${rung} --hits ${hits} --queries-fa ${queries} \
        --out spectrum-${rung}.json
    """
}

process SUMMARY {
    label 'process_medium'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path counts, stageAs: 'parts/count/?.json'
    path units, stageAs: 'parts/units/*'
    path kmers, stageAs: 'parts/kmers/*'
    path spectra, stageAs: 'parts/spectrum/*'

    output:
    path 'scale-profile.json'
    path 'check.tsv'

    script:
    """
    ${py()} summary --dir parts --source '${params.store}' --out scale-profile.json --check check.tsv
    """
}

workflow {
    if (!params.store) error 'Set --store to the release members buckets (a glob)'
    buckets = Channel.fromPath(params.store, checkIfExists: true)
    COUNT(buckets)
    counts = COUNT.out.collect()
    TOTALS(counts)
    totals = TOTALS.out.first()
    SAMPLE(buckets, totals)
    samples = SAMPLE.out.rung.collect()
    rungs = Channel.fromList(params.rungs.tokenize(',')*.trim())
    UNITS(rungs, samples, totals)
    ks = Channel.fromList(params.ks.tokenize(',')*.trim())
    KMERS(UNITS.out.combine(ks), samples, totals)
    QUERIES(SAMPLE.out.queries.collect())
    queries = QUERIES.out.first()
    FASTA(rungs, samples, totals)
    ALIGN(FASTA.out, queries)
    SPECTRUM(ALIGN.out, samples, totals, queries)
    SUMMARY(counts, UNITS.out.map { it[1] }.collect(), KMERS.out.collect(), SPECTRUM.out.collect())
}
