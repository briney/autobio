# autobio-baddg — BA-ddG binding ddG prediction
#
# Boltzmann-aligned inverse folding model for predicting binding stability
# changes (ddG) from mutations in protein-protein complexes. Uses a
# thermodynamic cycle with a modified ProteinMPNN architecture.
#
# Reference: Li et al. (ICLR 2025). https://arxiv.org/abs/2410.09543
#
# Build context: containers/
#   docker build -f containers/baddg/Dockerfile containers/ -t autobio-baddg:1.0.0

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# --- Autobio protocol dependencies ------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends jq bc git \
    && rm -rf /var/lib/apt/lists/*

# --- BA-ddG source -----------------------------------------------------------
# Clone at a pinned commit for reproducibility.
RUN git clone https://github.com/aim-uofa/BA-DDG.git /app/baddg \
    && cd /app/baddg \
    && git checkout dd677025c0fd504c69e094118633ea2a88359067

# Patch BioPython compatibility: three_to_one was removed in BioPython >= 1.82.
# The upstream parsers.py imports it; replace with the dict-based equivalent.
RUN sed -i 's/from Bio.PDB.Polypeptide import three_to_one/from Bio.Data.PDBData import protein_letters_3to1 as _3to1; three_to_one = lambda s: _3to1[s]/' \
    /app/baddg/training/common_utils/protein/parsers.py

# --- Python dependencies ----------------------------------------------------
# gdown is needed to download model weights from Google Drive.
# easydict is required by BA-ddG for config handling.
RUN pip install --no-cache-dir \
    gdown \
    biopython \
    scipy \
    pandas \
    tqdm \
    easydict \
    pyyaml \
    scikit-learn

# --- Model weights -----------------------------------------------------------
# Download from Google Drive. Two checkpoints are needed:
#   1. ProteinMPNN backbone (soluble model, v_48_020)
#   2. Fine-tuned BA-ddG weights (3-fold CV checkpoint)
RUN mkdir -p /app/baddg/ckpt/soluble_model_weights \
    && gdown --fuzzy "https://drive.google.com/uc?id=1fTy-Z686onowsyv3jYmUQrvsDTZm1CRU" \
       -O /app/baddg/ckpt/soluble_model_weights/v_48_020.pt \
    && gdown --fuzzy "https://drive.google.com/uc?id=1Ln44fS_4OJcJjnTveO-UDG_5sDBoF-5-" \
       -O /app/baddg/ckpt/ddg_model.ckpt

# Verify weights were downloaded successfully
RUN test -s /app/baddg/ckpt/soluble_model_weights/v_48_020.pt \
    && test -s /app/baddg/ckpt/ddg_model.ckpt \
    && echo "Model weights verified." \
    || (echo "ERROR: Model weight download failed" && exit 1)

# --- PYTHONPATH for BA-ddG imports -------------------------------------------
# The upstream repo has no setup.py; modules are imported directly from training/
ENV PYTHONPATH="/app/baddg/training:${PYTHONPATH}"

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY baddg/validate_config.sh /opt/tool/validate_config.sh
COPY baddg/run.sh             /opt/tool/run.sh
COPY baddg/standardize.sh     /opt/tool/standardize.sh
COPY baddg/standardize.py     /opt/tool/standardize.py
COPY baddg/inference.py       /opt/tool/inference.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
