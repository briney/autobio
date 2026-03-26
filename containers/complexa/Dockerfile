# autobio-complexa — Proteina-Complexa binder design
#
# Contains all three model variants: protein binder, ligand binder, and AME.
# Model weights are baked in at build time for a fully self-contained image.
#
# Multi-stage build:
#   Generate-only (smaller, faster):
#     docker build -f containers/complexa/Dockerfile --target generate -t autobio-complexa:2.0.0-generate .
#   Full pipeline (default — includes AF2, RF3, ESM2, MPNN for evaluation):
#     docker build -f containers/complexa/Dockerfile -t autobio-complexa:2.0.0 .
#
# The build context is the repo root because the upstream UV build script
# needs the full proteina-complexa clone.

# =============================================================================
# Stage: base — system deps, clone, build Python environment
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.08-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["bash", "-lc"]

# --- System dependencies -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq bc rsync libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# --- Clone Proteina-Complexa -------------------------------------------------
WORKDIR /app/proteina-complexa
RUN git clone --depth=1 https://github.com/NVIDIA-Digital-Bio/proteina-complexa.git .

# --- Build Python environment via upstream UV script --------------------------
# This installs PyTorch 2.7.0+cu126, JAX+CUDA, ColabFold, tmol, foundry, and
# all other deps. The full env is built even for generate-only so both stages
# share this expensive layer.
RUN bash env/build_uv_env.sh --root /app

# =============================================================================
# Stage: generate — Complexa checkpoints only (generate-only image)
# =============================================================================
FROM base AS generate

# --- Download all three Complexa checkpoint variants --------------------------
# Weights are fetched directly from NVIDIA NGC (no API key required).
ENV COMPLEXA_WEIGHTS=/app/proteina-complexa/ckpts
RUN mkdir -p "$COMPLEXA_WEIGHTS" && \
    NGC_BASE_PROTEIN="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/proteina_complexa/1.0/files?redirect=true&path=" && \
    NGC_BASE_LIGAND="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/proteina_complexa_ligand/1.0/files?redirect=true&path=" && \
    NGC_BASE_AME="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/proteina_complexa_ame/1.0/files?redirect=true&path=" && \
    echo "Downloading protein binder weights..." && \
    wget -q --show-progress -O "$COMPLEXA_WEIGHTS/complexa.ckpt" "${NGC_BASE_PROTEIN}complexa.ckpt" && \
    wget -q --show-progress -O "$COMPLEXA_WEIGHTS/complexa_ae.ckpt" "${NGC_BASE_PROTEIN}complexa_ae.ckpt" && \
    echo "Downloading ligand binder weights..." && \
    wget -q --show-progress -O "$COMPLEXA_WEIGHTS/complexa_ligand.ckpt" "${NGC_BASE_LIGAND}complexa_ligand.ckpt" && \
    wget -q --show-progress -O "$COMPLEXA_WEIGHTS/complexa_ligand_ae.ckpt" "${NGC_BASE_LIGAND}complexa_ligand_ae.ckpt" && \
    echo "Downloading AME weights..." && \
    wget -q --show-progress -O "$COMPLEXA_WEIGHTS/complexa_ame.ckpt" "${NGC_BASE_AME}complexa_ame.ckpt" && \
    wget -q --show-progress -O "$COMPLEXA_WEIGHTS/complexa_ame_ae.ckpt" "${NGC_BASE_AME}complexa_ame_ae.ckpt" && \
    echo "All Complexa weights downloaded."

# --- Set environment for the UV venv -----------------------------------------
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/proteina-complexa/src"
# Silence missing community model env vars (not needed for generate-only)
ENV AF2_DIR=""
ENV RF3_CKPT_PATH=""
ENV RF3_EXEC_PATH=""

# --- Autobio protocol scripts ------------------------------------------------
WORKDIR /workspace
COPY containers/base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

COPY containers/complexa/validate_config.sh /opt/tool/validate_config.sh
COPY containers/complexa/run.sh             /opt/tool/run.sh
COPY containers/complexa/run_complexa.py    /opt/tool/run_complexa.py
COPY containers/complexa/standardize.sh     /opt/tool/standardize.sh
COPY containers/complexa/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]

# =============================================================================
# Stage: full — community model weights for the design pipeline (default)
# =============================================================================
FROM generate AS full

# --- AlphaFold2 parameters (~5 GB) -------------------------------------------
RUN mkdir -p /app/proteina-complexa/community_models/ckpts/AF2 && \
    echo "Downloading AlphaFold2 parameters..." && \
    wget -q --show-progress -O /tmp/alphafold_params.tar \
        "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" && \
    tar -xf /tmp/alphafold_params.tar -C /app/proteina-complexa/community_models/ckpts/AF2/ && \
    rm /tmp/alphafold_params.tar && \
    echo "AF2 parameters downloaded."

# --- RoseTTAFold3 checkpoint (~2.5 GB) ---------------------------------------
RUN mkdir -p /app/proteina-complexa/community_models/ckpts/RF3 && \
    echo "Downloading RoseTTAFold3 checkpoint..." && \
    wget -q --show-progress -O \
        /app/proteina-complexa/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt \
        "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt" && \
    echo "RF3 checkpoint downloaded."

# --- ESM2 (facebook/esm2_t33_650M_UR50D, ~2.6 GB) ---------------------------
# Uses the upstream download script which calls transformers from_pretrained().
RUN echo "Downloading ESM2 model..." && \
    python /app/proteina-complexa/script_utils/download/download_esm_model.py \
        --cache-dir /app/proteina-complexa/community_models/ckpts/ESM2/ && \
    echo "ESM2 model downloaded."

# --- ProteinMPNN weights (~50 MB) --------------------------------------------
RUN MPNN_BASE="https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57" && \
    CA_DIR="/app/proteina-complexa/community_models/ProteinMPNN/ca_model_weights" && \
    VAN_DIR="/app/proteina-complexa/community_models/ProteinMPNN/vanilla_model_weights" && \
    mkdir -p "$CA_DIR" "$VAN_DIR" && \
    echo "Downloading ProteinMPNN weights..." && \
    for f in v_48_002.pt v_48_010.pt v_48_020.pt; do \
        wget -q -O "$CA_DIR/$f" "$MPNN_BASE/ca_model_weights/$f"; \
    done && \
    for f in v_48_002.pt v_48_010.pt v_48_020.pt v_48_030.pt; do \
        wget -q -O "$VAN_DIR/$f" "$MPNN_BASE/vanilla_model_weights/$f"; \
    done && \
    echo "ProteinMPNN weights downloaded."

# --- LigandMPNN weights (~500 MB) --------------------------------------------
RUN LMPNN_DIR="/app/proteina-complexa/community_models/LigandMPNN/model_params" && \
    LMPNN_BASE="https://files.ipd.uw.edu/pub/ligandmpnn" && \
    mkdir -p "$LMPNN_DIR" && \
    echo "Downloading LigandMPNN weights..." && \
    for f in \
        proteinmpnn_v_48_002.pt proteinmpnn_v_48_010.pt \
        proteinmpnn_v_48_020.pt proteinmpnn_v_48_030.pt \
        ligandmpnn_v_32_005_25.pt ligandmpnn_v_32_010_25.pt \
        ligandmpnn_v_32_020_25.pt ligandmpnn_v_32_030_25.pt \
        per_residue_label_membrane_mpnn_v_48_020.pt \
        global_label_membrane_mpnn_v_48_020.pt \
        solublempnn_v_48_002.pt solublempnn_v_48_010.pt \
        solublempnn_v_48_020.pt solublempnn_v_48_030.pt \
        ligandmpnn_sc_v_32_002_16.pt; do \
        wget -q -O "$LMPNN_DIR/$f" "$LMPNN_BASE/$f"; \
    done && \
    echo "LigandMPNN weights downloaded."

# --- Override environment for full pipeline -----------------------------------
ENV AF2_DIR=/app/proteina-complexa/community_models/ckpts/AF2
ENV RF3_CKPT_PATH=/app/proteina-complexa/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt
ENV RF3_EXEC_PATH=/app/.venv/bin/rf3
