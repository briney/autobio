# autobio-esm-if1 — ESM-IF1 inverse folding and sequence scoring
#
# Uses Facebook Research's ESM-IF1 (esm_if1_gvp4_t16_142M_UR50) for:
#   - Designing protein sequences from backbone structures (inverse folding)
#   - Scoring sequences against structures (conditional log-likelihood)
#
# Build context: containers/
#   docker build -f containers/esm-if1/Dockerfile containers/ -t autobio-esm-if1:1.0.0

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
    && rm -rf /var/lib/apt/lists/*

# --- Python packages ---------------------------------------------------------
# ESM-IF1's GVP modules require torch_scatter/torch_sparse from PyG.
# Install torch first, then PyG extensions from the matching wheel index,
# then fair-esm and biotite.
RUN pip3 install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
       torch_geometric \
    && pip3 install --no-cache-dir \
       torch_scatter torch_sparse \
       -f https://data.pyg.org/whl/torch-2.5.0+cu121.html \
    && pip3 install --no-cache-dir \
       fair-esm "biotite<1.0"

# --- Download model weights --------------------------------------------------
# Place weights where torch.hub expects them so esm.pretrained loads offline.
ENV TORCH_HOME=/app/esm_if1
RUN mkdir -p "$TORCH_HOME/hub/checkpoints" \
    && wget -q -O "$TORCH_HOME/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt" \
       https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt

# --- Patch fair-esm for GPU device placement ---------------------------------
# fair-esm 2.0.0 has bugs where tensors are created on CPU inside model.sample()
# and CoordBatchConverter, causing device mismatches when the model is on GPU.
COPY esm-if1/patch_esm_device.py /opt/tool/patch_esm_device.py
RUN python3 /opt/tool/patch_esm_device.py

# Verify the model loads correctly (catches import and weight-path issues)
RUN python3 -c "import esm; model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50(); print('ESM-IF1 loaded successfully')"

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific scripts ---------------------------------------------------
COPY esm-if1/validate_config.sh /opt/tool/validate_config.sh
COPY esm-if1/run.sh             /opt/tool/run.sh
COPY esm-if1/run_esm_if1.py     /opt/tool/run_esm_if1.py
COPY esm-if1/standardize.sh     /opt/tool/standardize.sh
COPY esm-if1/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
