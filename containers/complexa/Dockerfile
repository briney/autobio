# autobio-complexa — Proteina-Complexa binder design (generate-only)
#
# Contains all three model variants: protein binder, ligand binder, and AME.
# Model weights are baked in at build time for a fully self-contained image.
#
# Build from repo root:
#   docker build -f containers/complexa/Dockerfile -t autobio-complexa:1.0.0 .
#
# The build context is the repo root because the upstream UV build script
# needs the full proteina-complexa clone.

FROM nvcr.io/nvidia/pytorch:24.08-py3

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
# This installs PyTorch 2.7.0+cu126, all deps, and the proteinfoundation package.
RUN bash env/build_uv_env.sh --root /app

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
    echo "All weights downloaded."

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
