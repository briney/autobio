# autobio-antifold — AntiFold antibody inverse folding and scoring
#
# AntiFold is an antibody-specific inverse folding model fine-tuned from
# ESM-IF1, trained on solved/predicted antibody structures from SAbDab/OAS.
# Includes ANARCI for automatic IMGT renumbering of input PDBs.
#
# Build context: containers/
#   docker build -f containers/antifold/Dockerfile containers/ -t antifold:1.0.0

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System packages ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        jq \
        bc \
        python3 \
        python3-pip \
        wget \
        git \
        hmmer \
    && rm -rf /var/lib/apt/lists/*

# --- Python packages ---------------------------------------------------------
# Install PyTorch with CUDA 12.1 support first, then AntiFold and ANARCI.
RUN pip3 install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cu121

# Install AntiFold from GitHub (includes ESM-IF1 dependencies)
RUN pip3 install --no-cache-dir \
       git+https://github.com/oxpig/AntiFold.git

# Install ANARCI for IMGT renumbering
RUN pip3 install --no-cache-dir anarci

# Set up ANARCI HMM databases (required for numbering)
RUN python3 -c "from anarci import anarci; print('ANARCI loaded successfully')"

# --- Verify model loads ------------------------------------------------------
RUN python3 -c "from antifold.main import load_model; m = load_model(); print('AntiFold loaded successfully')"

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific scripts ---------------------------------------------------
COPY antifold/validate_config.sh /opt/tool/validate_config.sh
COPY antifold/run.sh             /opt/tool/run.sh
COPY antifold/run_antifold.py    /opt/tool/run_antifold.py
COPY antifold/standardize.sh     /opt/tool/standardize.sh
COPY antifold/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
