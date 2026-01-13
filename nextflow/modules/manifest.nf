process MAKE_MANIFEST {
  tag 'samples'

  input:
    path samples_yaml

  output:
    path 'samples.tsv'

  script:
  """
  python /opt/triton/nf/bin/make_manifest.py \
    --samples-yaml ${samples_yaml} \
    --out samples.tsv
  """
}
