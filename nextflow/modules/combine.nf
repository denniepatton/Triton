process COMBINE_FMS {
  tag 'combine'

  publishDir { "${params.outdir}" }, mode: 'copy', overwrite: true

  input:
    path sample_dirs

  output:
    path 'TritonCompositeFM.tsv'

  script:
  """
  python /opt/triton/Triton/triton_cleanup.py \
    --results_dir . \
    --output_format ${params.output_format}
  """
}
