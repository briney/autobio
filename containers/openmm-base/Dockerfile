# autobio-openmm-base — OpenMM molecular simulation toolkit
#
# Base image for all OpenMM-derived tool containers. NOT run directly —
# derived images (openmm-amber-minimize, etc.) add tool-specific scripts
# and set the autobio entrypoint.
#
# Includes CUDA-capable OpenMM. All derived containers require GPU —
# the default platform is CUDA.
#
# Build context: containers/
#   docker build -f containers/openmm-base/Dockerfile containers/ -t autobio-openmm-base:8.5.0

FROM condaforge/miniforge3:latest

# --- Autobio protocol dependencies (jq, bc required by base-entrypoint.sh) ---
RUN apt-get update && apt-get install -y --no-install-recommends jq bc \
    && rm -rf /var/lib/apt/lists/*

# --- OpenMM 8.5 with CUDA support + PDBFixer for structure cleanup ---
# --- OpenMM 8.5 with CUDA support + PDBFixer for structure cleanup ---
# Requires NVIDIA driver 575+ (CUDA 12.9 support) on the host.
RUN mamba install -y -c conda-forge \
        openmm=8.5.0 \
        pdbfixer \
    && mamba clean -afy

# --- Shared OpenMM utilities used by all derived containers ----------------
COPY openmm-base/openmm_utils.py /opt/openmm/openmm_utils.py
ENV PYTHONPATH="/opt/openmm"

# Default to GPU platform — all derived containers require CUDA
ENV OPENMM_DEFAULT_PLATFORM=CUDA

# No ENTRYPOINT — this image is a base for derived tool images
