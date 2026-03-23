# autobio-boltz — Boltz biomolecular structure prediction (Boltz-1 and Boltz-2)
#
# Extends an NVIDIA CUDA runtime image with Boltz and pre-cached model weights.
# Supports both Boltz-1 and Boltz-2 — the config.json written by the host-side
# runner determines which model is used.
#
# Build context: containers/
#   docker build -f containers/boltz/Dockerfile containers/ -t autobio-boltz:1.0.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- System dependencies ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv jq bc wget \
    && rm -rf /var/lib/apt/lists/*

# --- Install PyTorch with CUDA 12.8 support first ----------------------------
# Pin a PyTorch version compatible with the host GPU driver. boltz[cuda] would
# pull the latest PyTorch which may target a newer CUDA than the host supports.
RUN pip3 install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# --- Install Boltz + CUDA acceleration extras ---------------------------------
# Install boltz first (PyTorch already satisfied above), then the CUDA extras
# separately. Using boltz[cuda] directly would re-install PyTorch.
RUN pip3 install --no-cache-dir boltz \
    cuequivariance_torch cuequivariance_ops_cu12 cuequivariance_ops_torch_cu12

# --- Download model weights at build time (self-contained image) ---------------
ENV BOLTZ_CACHE=/app/boltz/cache
RUN python3 -c "\
from pathlib import Path; \
from boltz.main import download_boltz1, download_boltz2; \
cache = Path('/app/boltz/cache'); \
cache.mkdir(parents=True, exist_ok=True); \
download_boltz1(cache); \
download_boltz2(cache)"

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY boltz/validate_config.sh /opt/tool/validate_config.sh
COPY boltz/run.sh             /opt/tool/run.sh
COPY boltz/standardize.sh     /opt/tool/standardize.sh
COPY boltz/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
