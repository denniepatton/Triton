# Triton container image for Nextflow execution
# Built from conda environment definition in containers/environment.yml
#
# This image is designed to be pulled as an OCI image from a registry (e.g. GHCR)
# and run via Docker (local) or Singularity/Apptainer (HPC).

FROM mambaorg/micromamba:1.5.10

USER root

# Keep the image lean and deterministic
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER containers/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
  && micromamba clean -a -y

# Install Triton code and Nextflow helper scripts into the image so jobs don't depend
# on the host filesystem being mounted inside the container.
WORKDIR /opt/triton
COPY --chown=$MAMBA_USER:$MAMBA_USER Triton/ /opt/triton/Triton/
COPY --chown=$MAMBA_USER:$MAMBA_USER nc_fitting/ /opt/triton/nc_fitting/
COPY --chown=$MAMBA_USER:$MAMBA_USER nf/bin/ /opt/triton/nf/bin/

# No entrypoint: Nextflow runs process scripts directly.
ENTRYPOINT []
CMD ["bash"]

USER $MAMBA_USER
