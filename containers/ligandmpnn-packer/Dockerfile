# autobio-ligandmpnn-packer — LigandMPNN sidechain packing for mutant building
#
# Uses LigandMPNN's sidechain packing neural network (Packer model) to
# predict chi angles and produce full-atom mutant structures.
# The Rosetta Commons foundry does NOT expose sidechain packing, so this
# container is built directly from the original dauparas/LigandMPNN code.
#
# Build context: containers/
#   docker build -f containers/ligandmpnn-packer/Dockerfile containers/ -t ligandmpnn-packer:1.0.0

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System packages ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        jq \
        bc \
        python3 \
        python3-pip \
        wget \
    && rm -rf /var/lib/apt/lists/*

# --- Python packages ---------------------------------------------------------
# --- Python packages ---------------------------------------------------------
# ProDy needs C compiler + Python headers to build its C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev \
    && pip3 install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
       numpy ml-collections dm-tree biopython ProDy scipy \
    && apt-get purge -y build-essential python3-dev && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# --- Clone LigandMPNN at pinned commit --------------------------------------
RUN git clone https://github.com/dauparas/LigandMPNN.git /app/LigandMPNN \
    && cd /app/LigandMPNN \
    && git checkout 26ec57ac976ade5379920dbd43c7f97a91cf82de

# --- Download model checkpoints ---------------------------------------------
RUN mkdir -p /app/LigandMPNN/model_params \
    && wget -q -O /app/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
       https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt \
    && wget -q -O /app/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt \
       https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific scripts ---------------------------------------------------
COPY ligandmpnn-packer/validate_config.sh /opt/tool/validate_config.sh
COPY ligandmpnn-packer/run.sh             /opt/tool/run.sh
COPY ligandmpnn-packer/standardize.sh     /opt/tool/standardize.sh
COPY ligandmpnn-packer/standardize.py     /opt/tool/standardize.py
COPY ligandmpnn-packer/apply_mutations.py /opt/tool/apply_mutations.py
COPY ligandmpnn-packer/pack_sidechains.py /opt/tool/pack_sidechains.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
