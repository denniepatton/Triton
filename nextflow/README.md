# Triton (Nextflow)

This directory contains a fresh Nextflow (DSL2) implementation that reuses Triton’s existing YAML configs:

- `config/samples.yaml` (sample → BAM/CRAM + GC-bias)
- `config/config.yaml` (annotation, reference genome, run mode, etc.)

The workflow is container-first and intended to run with:

- Nextflow `<= 24.04.3`
- Singularity `<= 3.5.3` **or** Apptainer `<= 1.1.6`
- Docker (for local builds/runs)

## Quick start (local with Docker)

1) Build the container locally:

```bash
docker build -f containers/Dockerfile -t triton:dev .
```

2) Run Nextflow (from repo root):

```bash
nextflow run nf -profile docker \
  --container triton:dev \
  --samples_yaml config/samples.yaml \
  --config_yaml  config/config.yaml \
  --outdir results
```

Outputs:
- `results/<sample>/<sample>_TritonFeatures.tsv`
- `results/<sample>/<sample>_TritonSignalProfiles.npz`
- `results/<sample>/<sample>_TritonFragmentationProfiles.npz`
- `results/TritonCompositeFM.tsv`

## HPC (Slurm + Apptainer/Singularity)

Assuming the image is published (see below), run from repo root:

Apptainer:
```bash
nextflow run nf -profile slurm_apptainer \
  --container ghcr.io/denniepatton/triton:latest \
  --bind_paths '/fh,/hpc,/scratch' \
  --queue campus-new
```

Singularity:
```bash
nextflow run nf -profile slurm_singularity \
  --container ghcr.io/denniepatton/triton:latest \
  --bind_paths '/fh,/hpc,/scratch' \
  --queue campus-new
```

Notes:
- Your `samples.yaml` and `config.yaml` can keep absolute paths (e.g. `/fh/...`), but those paths must be bind-mounted into the container.
- For CRAM inputs, `reference_genome` must also be accessible on the compute nodes.

## Publishing the container (recommended: GHCR)

This repo includes a GitHub Actions workflow at `.github/workflows/docker-build.yml` that builds and pushes the image to GHCR on pushes/tags.

Once the repo is public, you can:

- Create a tag, e.g. `v2.0.0`, and push it
- The workflow will push `ghcr.io/<owner>/triton:<tag>` and `:latest` (default branch)

Pulling on HPC:

```bash
apptainer pull triton.sif docker://ghcr.io/denniepatton/triton:latest
```

Then run Nextflow with `--container docker://ghcr.io/...` if your site requires the explicit prefix.

## Parameters

Common overrides (without editing YAML):
- `--outdir <dir>`
- `--output_format long|wide`
- `--bind_paths '<comma-separated host paths>'`

Advanced overrides (if you don’t want to edit `config/config.yaml`):
- `--annotation <path>`
- `--reference_genome <path>`
- `--run_mode region|window|composite-window`
- `--map_quality <int>`
- `--size_range <min> <max>`

