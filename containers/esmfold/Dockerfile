# autobio-esmfold — ESMFold single-sequence protein structure prediction
#
# Predicts 3D protein structure from amino acid sequence using the ESM-2
# language model backbone with a folding trunk. No MSA or templates needed.
#
# Build context: containers/
#   docker build -f containers/esmfold/Dockerfile containers/ -t autobio-esmfold:1.0.0

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

# --- Download and cache ESMFold weights at build time -------------------------
ENV HF_HOME=/app/esmfold/hf_cache
RUN python3 -c "\
from transformers import EsmForProteinFolding, EsmTokenizer; \
EsmForProteinFolding.from_pretrained('facebook/esmfold_v1'); \
EsmTokenizer.from_pretrained('facebook/esmfold_v1')"
ENV TRANSFORMERS_OFFLINE=1

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY esmfold/validate_config.sh /opt/tool/validate_config.sh
COPY esmfold/run.sh             /opt/tool/run.sh
COPY esmfold/run_esmfold.py     /opt/tool/run_esmfold.py
COPY esmfold/standardize.sh     /opt/tool/standardize.sh
COPY esmfold/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
