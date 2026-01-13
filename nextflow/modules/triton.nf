process RUN_TRITON {
  tag { sample }

  publishDir { "${params.outdir}" }, mode: 'copy', overwrite: true

  input:
    tuple val(sample), val(bam_path), val(gc_bias_path)
    path config_yaml
    path config_dir

  output:
    path "${sample}" , emit: sample_dir

  script:
  """
  python /opt/triton/nf/bin/run_triton_from_yaml.py \
    --repo-dir /opt/triton \
    --config-yaml ${config_yaml} \
    --sample ${sample} \
    --bam ${bam_path} \
    --gc-bias ${gc_bias_path} \
    --cpus ${task.cpus} \
    --results-dir .
  """
}
