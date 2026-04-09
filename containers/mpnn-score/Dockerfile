# autobio-mpnn-score — ProteinMPNN and LigandMPNN sequence scoring
#
# Uses the official dauparas/LigandMPNN Python API to compute conditional
# log-likelihoods.  The Rosetta Commons foundry CLI does NOT expose scoring,
# so this container is built directly from the original LigandMPNN code.
#
# Build context: containers/
#   docker build -f containers/mpnn-score/Dockerfile containers/ -t mpnn-score:1.0.0

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
    && wget -q -O /app/LigandMPNN/model_params/proteinmpnn_v_48_020.pt \
       https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt \
    && wget -q -O /app/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
       https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific scripts ---------------------------------------------------
COPY mpnn-score/validate_config.sh  /opt/tool/validate_config.sh
COPY mpnn-score/run.sh              /opt/tool/run.sh
COPY mpnn-score/score_sequences.py  /opt/tool/score_sequences.py
COPY mpnn-score/standardize.sh      /opt/tool/standardize.sh
COPY mpnn-score/standardize.py      /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
