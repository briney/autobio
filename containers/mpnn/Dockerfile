# autobio-mpnn — ProteinMPNN and LigandMPNN inverse folding
#
# Extends the official rosettacommons/foundry image with the autobio
# three-phase container protocol and downloads default checkpoints.
#
# Build context: containers/
#   docker build -f containers/mpnn/Dockerfile containers/ -t autobio-mpnn:1.0.0

FROM rosettacommons/foundry:latest

# --- Autobio protocol dependencies -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends jq bc wget \
    && rm -rf /var/lib/apt/lists/*

# --- Download default checkpoints -------------------------------------------
RUN mkdir -p /app/foundry/checkpoints \
    && wget -q -O /app/foundry/checkpoints/proteinmpnn_v_48_020.pt \
       https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt \
    && wget -q -O /app/foundry/checkpoints/ligandmpnn_v_32_010_25.pt \
       https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY mpnn/validate_config.sh /opt/tool/validate_config.sh
COPY mpnn/run.sh             /opt/tool/run.sh
COPY mpnn/standardize.sh     /opt/tool/standardize.sh
COPY mpnn/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
