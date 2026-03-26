# autobio-chai — Chai-1 multi-modal structure prediction
#
# Extends an NVIDIA CUDA runtime image with Chai-1 and pre-cached model weights.
# Supports proteins, ligands, DNA, RNA, glycans with restraints and covalent bonds.
#
# Build context: containers/
#   docker build -f containers/chai/Dockerfile containers/ -t autobio-chai:1.0.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- System dependencies ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv jq bc wget \
    && rm -rf /var/lib/apt/lists/*

# --- Install PyTorch with CUDA 12.8 support first ----------------------------
# Pin a PyTorch version compatible with the host GPU driver. chai-lab requires
# torch>=2.3.1 with bfloat16 support (Ampere+ GPUs).
RUN pip3 install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# --- Install Chai-1 ----------------------------------------------------------
# PyTorch already satisfied above; chai-lab pulls in rdkit, einops, gemmi, etc.
RUN pip3 install --no-cache-dir chai_lab==0.6.1

# --- Download model weights at build time (self-contained image) ---------------
ENV CHAI_DOWNLOADS_DIR=/app/chai/downloads
RUN mkdir -p /app/chai/downloads/models_v2 \
    && cd /app/chai/downloads/models_v2 \
    && for model in feature_embedding.pt token_embedder.pt trunk.pt \
                    diffusion_module.pt confidence_head.pt bond_loss_input_proj.pt; do \
           echo "Downloading $model..." \
           && wget -q "https://chaiassets.com/chai1-inference-depencencies/models_v2/$model"; \
       done \
    && cd /app/chai/downloads \
    && echo "Downloading conformers_v1.apkl..." \
    && wget -q "https://chaiassets.com/chai1-inference-depencencies/conformers_v1.apkl"

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY chai/validate_config.sh /opt/tool/validate_config.sh
COPY chai/run.sh             /opt/tool/run.sh
COPY chai/run_chai.py        /opt/tool/run_chai.py
COPY chai/standardize.sh     /opt/tool/standardize.sh
COPY chai/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
