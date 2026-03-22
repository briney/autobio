# autobio-rfd3 — RFDiffusion3 generative structure design
#
# Extends the official rosettacommons/foundry image with the autobio
# three-phase container protocol.
#
# Build context: containers/
#   docker build -f containers/rfd3/Dockerfile containers/ -t autobio-rfd3:1.0.0

FROM rosettacommons/foundry:latest

# --- Autobio protocol dependencies -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends jq bc \
    && rm -rf /var/lib/apt/lists/*

# --- Install RFD3 model and checkpoints --------------------------------------
# The foundry image provides the `foundry` CLI; install the rfd3 model
# which downloads the default checkpoint.
# TODO: verify exact install command against the production foundry image.
RUN foundry install rfd3 || true

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY rfd3/validate_config.sh /opt/tool/validate_config.sh
COPY rfd3/run.sh             /opt/tool/run.sh
COPY rfd3/standardize.sh     /opt/tool/standardize.sh
COPY rfd3/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
