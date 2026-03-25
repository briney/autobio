# autobio-esm — ESM-1b and ESM-2 protein language model embeddings
#
# Supports both ESM-1b (650M) and ESM-2 (8M, 35M, 150M, 650M by default).
# For ESM-2 3B/15B, build with --build-arg ESM2_EXTRA_CHECKPOINTS="3B" or "15B".
#
# Build context: containers/
#   docker build -f containers/esm/Dockerfile containers/ -t autobio-esm:1.0.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- System dependencies ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv jq bc wget \
    && rm -rf /var/lib/apt/lists/*

# --- Install PyTorch with CUDA 12.8 support -----------------------------------
RUN pip3 install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# --- Install HuggingFace transformers + dependencies --------------------------
RUN pip3 install --no-cache-dir transformers numpy safetensors accelerate

# --- Download and cache model weights at build time ----------------------------
# Pre-populate the HuggingFace cache so the container is fully self-contained.
# At runtime, TRANSFORMERS_OFFLINE=1 prevents any network requests.
ENV HF_HOME=/app/esm/hf_cache

# ESM-1b (650M)
RUN python3 -c "\
from transformers import EsmModel, EsmTokenizer; \
EsmModel.from_pretrained('facebook/esm1b_t33_650M_UR50S'); \
EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')"

# ESM-2: 8M, 35M, 150M, 650M (default set) — separate layers for better caching
RUN python3 -c "from transformers import EsmModel, EsmTokenizer; EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D'); EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')"
RUN python3 -c "from transformers import EsmModel, EsmTokenizer; EsmModel.from_pretrained('facebook/esm2_t12_35M_UR50D'); EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')"
RUN python3 -c "from transformers import EsmModel, EsmTokenizer; EsmModel.from_pretrained('facebook/esm2_t30_150M_UR50D'); EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')"
RUN python3 -c "from transformers import EsmModel, EsmTokenizer; EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D'); EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')"

# Optional: additional ESM-2 checkpoints (3B, 15B) via build arg
# Usage: docker build --build-arg ESM2_EXTRA_CHECKPOINTS="3B" ...
#    or: docker build --build-arg ESM2_EXTRA_CHECKPOINTS="3B,15B" ...
ARG ESM2_EXTRA_CHECKPOINTS=""
RUN if [ -n "$ESM2_EXTRA_CHECKPOINTS" ]; then \
        for ckpt in $(echo "$ESM2_EXTRA_CHECKPOINTS" | tr ',' ' '); do \
            case "$ckpt" in \
                3B) MODEL="facebook/esm2_t36_3B_UR50D" ;; \
                15B) MODEL="facebook/esm2_t48_15B_UR50D" ;; \
                *) echo "Unknown checkpoint: $ckpt" >&2; exit 1 ;; \
            esac; \
            python3 -c "from transformers import EsmModel, EsmTokenizer; EsmModel.from_pretrained('$MODEL'); EsmTokenizer.from_pretrained('$MODEL')"; \
        done; \
    fi
ENV TRANSFORMERS_OFFLINE=1

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY esm/validate_config.sh /opt/tool/validate_config.sh
COPY esm/run.sh             /opt/tool/run.sh
COPY esm/run_esm.py         /opt/tool/run_esm.py
COPY esm/standardize.sh     /opt/tool/standardize.sh
COPY esm/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
