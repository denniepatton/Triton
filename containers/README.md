# Triton container build + publish

This repo builds a container image from `containers/environment.yml` using `containers/Dockerfile`.

The recommended path is **GitHub Actions → GHCR**, because it avoids HPC network/proxy issues and produces a public OCI image that **Apptainer/Singularity** can pull via `docker://...`.

## Option A (recommended): GitHub Actions → GHCR

Prereqs:
- Repo on GitHub
- Default branch is `main`
- Actions enabled

Steps:
1. Commit + push the Dockerfile and environment:
   - `containers/Dockerfile`
   - `containers/environment.yml`

2. The workflow `.github/workflows/docker-build.yml` runs on:
   - pushes to `main`
   - tags like `v*` (recommended for versioned releases)

3. Create and push a release tag:

```bash
git tag v2.0.0
git push origin v2.0.0
```

4. After the workflow completes, the image will be available at:
- `ghcr.io/<owner>/triton:latest`
- `ghcr.io/<owner>/triton:v2.0.0`

5. Make the package public (needed for unauthenticated pulls):
- GitHub → your profile/org → **Packages** → `triton` → **Package settings** → **Change visibility** → Public

## Option B: Build + push manually with Docker

Use this if you can build on a machine that can reach conda channels.

1. Create a GitHub Personal Access Token (PAT) with:
- `write:packages` (to push)
- `read:packages` (to pull)

2. Login to GHCR:

```bash
export CR_PAT='...'
echo "$CR_PAT" | docker login ghcr.io -u <github-username> --password-stdin
```

3. Build and push:

```bash
docker build -f containers/Dockerfile -t ghcr.io/<owner>/triton:latest .
docker push ghcr.io/<owner>/triton:latest

# Optional version tag
docker tag ghcr.io/<owner>/triton:latest ghcr.io/<owner>/triton:v2.0.0
docker push ghcr.io/<owner>/triton:v2.0.0
```

## Pulling with Apptainer/Singularity

Apptainer:
```bash
apptainer pull triton.sif docker://ghcr.io/<owner>/triton:latest
```

Singularity (older sites):
```bash
singularity pull triton.sif docker://ghcr.io/<owner>/triton:latest
```

If the package is private, you’ll need registry auth (site-dependent). For Apptainer:

```bash
apptainer registry login --username <github-username> docker://ghcr.io
# password: use your PAT
```

## Nextflow usage

From repo root:

```bash
nextflow run nf -profile slurm_apptainer \
  --container ghcr.io/<owner>/triton:latest \
  --samples_yaml config/samples.yaml \
  --config_yaml  config/config.yaml \
  --bind_paths '/fh,/hpc,/scratch'
```

Notes:
- Absolute paths in your YAML must exist on compute nodes and be bind-mounted into the container.
- For CRAM, `reference_genome` must also be accessible/bound.

## Troubleshooting

- If Docker build fails during `micromamba install` with SSL/proxy errors, prefer **Option A** (GitHub Actions).
- If you must build behind a proxy, set build-time env vars (ask your HPC admins for exact values):
  - `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`
