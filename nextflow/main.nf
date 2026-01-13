#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * Triton Nextflow workflow (DSL2)
 * - Reads config/samples.yaml to obtain sample BAM/CRAM + GC bias paths
 * - Reads config/config.yaml for Triton parameters (annotation, reference, etc.)
 * - Runs one Triton job per sample
 * - Combines per-sample feature matrices into results/TritonCompositeFM.tsv
 */

include { MAKE_MANIFEST } from './modules/manifest'
include { RUN_TRITON }    from './modules/triton'
include { COMBINE_FMS }   from './modules/combine'

workflow {
    // Build a simple TSV manifest from samples.yaml so we can avoid YAML parsing in Groovy.
    manifest_ch = MAKE_MANIFEST( file(params.samples_yaml) )

    samples_ch = manifest_ch
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            def sample = row.sample as String
            def bam    = row.bam as String
            def gcBias = row.gc_bias as String
            tuple(sample, bam, gcBias)
        }

    sample_dirs_ch = RUN_TRITON(
        samples_ch,
        file(params.config_yaml),
        file(params.config_dir)
    )

    composite = COMBINE_FMS(sample_dirs_ch.collect())

    emit:
      composite
}
