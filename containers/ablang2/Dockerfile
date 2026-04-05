# autobio ablang2 — AbLang2 antibody language model embeddings and PLL
#
# Build context: containers/
#   docker build -f containers/ablang2/Dockerfile \
#       containers/ -t autobio-ablang2:1.0.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- System dependencies ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv jq bc wget \
    && rm -rf /var/lib/apt/lists/*

# --- Install PyTorch with CUDA 12.8 support -----------------------------------
RUN pip3 install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# --- Install AbLang2 and dependencies ----------------------------------------
RUN pip3 install --no-cache-dir ablang2 einops rotary-embedding-torch numpy

# --- Download and cache model weights at build time ---------------------------
# ablang2 stores weights inside its own package directory at
# <site-packages>/ablang2/model-weights-ablang2-paired/
RUN python3 -c "import ablang2; ablang2.pretrained('ablang2-paired')"

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY ablang2/validate_config.sh /opt/tool/validate_config.sh
COPY ablang2/run.sh             /opt/tool/run.sh
COPY ablang2/run_ablang2.py     /opt/tool/run_ablang2.py
COPY ablang2/standardize.sh     /opt/tool/standardize.sh
COPY ablang2/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
