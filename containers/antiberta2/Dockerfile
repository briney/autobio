# autobio antiberta2 — AntiBERTa2 antibody language model embeddings and PLL
#
# Build context: containers/
#   docker build -f containers/antiberta2/Dockerfile \
#       containers/ -t autobio-antiberta2:1.0.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- System dependencies ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv jq bc wget \
    && rm -rf /var/lib/apt/lists/*

# --- Install PyTorch with CUDA 12.8 support -----------------------------------
RUN pip3 install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# --- Install HuggingFace transformers + dependencies --------------------------
RUN pip3 install --no-cache-dir transformers numpy safetensors accelerate protobuf rjieba

# --- Download and cache model weights at build time ---------------------------
ENV HF_HOME=/app/antiberta2/hf_cache

RUN python3 -c "\
from transformers import RoFormerForMaskedLM, RoFormerTokenizer; \
RoFormerForMaskedLM.from_pretrained('alchemab/antiberta2'); \
RoFormerTokenizer.from_pretrained('alchemab/antiberta2')"

ENV TRANSFORMERS_OFFLINE=1

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY antiberta2/validate_config.sh /opt/tool/validate_config.sh
COPY antiberta2/run.sh             /opt/tool/run.sh
COPY antiberta2/run_antiberta2.py  /opt/tool/run_antiberta2.py
COPY antiberta2/standardize.sh     /opt/tool/standardize.sh
COPY antiberta2/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
