# autobio antibody-lm — Antibody language model embeddings and PLL
#
# Parameterized Dockerfile: build one image per model by setting MODEL_NAME.
#
# Build context: containers/
#   docker build -f containers/antibody-lm/Dockerfile \
#       --build-arg MODEL_NAME=brineylab/CurrAb \
#       containers/ -t autobio-currab:1.0.0
#
#   docker build -f containers/antibody-lm/Dockerfile \
#       --build-arg MODEL_NAME=brineylab/ft-ESM \
#       containers/ -t autobio-ft-esm:1.0.0
#
#   docker build -f containers/antibody-lm/Dockerfile \
#       --build-arg MODEL_NAME=brineylab/BALM-paired \
#       containers/ -t autobio-balm-paired:1.0.0
#
#   docker build -f containers/antibody-lm/Dockerfile \
#       --build-arg MODEL_NAME=brineylab/BALM-unpaired \
#       containers/ -t autobio-balm-unpaired:1.0.0

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
ARG MODEL_NAME
ENV HF_HOME=/app/antibody-lm/hf_cache

RUN if [ -z "$MODEL_NAME" ]; then echo "ERROR: MODEL_NAME build arg is required" >&2; exit 1; fi \
    && python3 -c "\
from transformers import AutoModelForMaskedLM, AutoTokenizer; \
AutoModelForMaskedLM.from_pretrained('${MODEL_NAME}'); \
AutoTokenizer.from_pretrained('${MODEL_NAME}')"

ENV TRANSFORMERS_OFFLINE=1

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY antibody-lm/validate_config.sh /opt/tool/validate_config.sh
COPY antibody-lm/run.sh             /opt/tool/run.sh
COPY antibody-lm/run_antibody_lm.py /opt/tool/run_antibody_lm.py
COPY antibody-lm/standardize.sh     /opt/tool/standardize.sh
COPY antibody-lm/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
