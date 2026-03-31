# autobio-openmm-base — OpenMM molecular simulation toolkit
#
# Base image for all OpenMM-derived tool containers. NOT run directly —
# derived images (openmm-amber-minimize, etc.) add tool-specific scripts
# and set the autobio entrypoint.
#
# Includes CUDA-capable OpenMM so derived images can run on GPU without
# rebuilding the base. The default platform is CPU; GPU containers can
# override OPENMM_DEFAULT_PLATFORM to CUDA.
#
# Build context: containers/
#   docker build -f containers/openmm-base/Dockerfile containers/ -t autobio-openmm-base:8.2.0

FROM condaforge/miniforge3:latest

# --- Autobio protocol dependencies (jq, bc required by base-entrypoint.sh) ---
RUN apt-get update && apt-get install -y --no-install-recommends jq bc \
    && rm -rf /var/lib/apt/lists/*

# --- OpenMM 8.2 with CUDA support + PDBFixer for structure cleanup ---
RUN mamba install -y -c conda-forge \
        openmm=8.2.0 \
        pdbfixer \
        cudatoolkit \
    && mamba clean -afy

# --- Shared OpenMM utilities used by all derived containers ----------------
COPY openmm-base/openmm_utils.py /opt/openmm/openmm_utils.py
ENV PYTHONPATH="/opt/openmm"

# Default to CPU platform — derived GPU containers can override to CUDA
ENV OPENMM_DEFAULT_PLATFORM=CPU

# No ENTRYPOINT — this image is a base for derived tool images
