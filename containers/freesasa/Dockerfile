# autobio-freesasa — FreeSASA SASA and buried surface area calculation
#
# Calculates solvent-accessible surface area (SASA) and buried surface area
# (BSA) at protein-protein interfaces. CPU-only, no GPU required.
#
# Build context: containers/
#   docker build -f containers/freesasa/Dockerfile containers/ -t autobio-freesasa:2.2.1

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# --- Autobio protocol dependencies + build tools for freesasa C extension ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq bc build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- FreeSASA (C extension, requires gcc to build) --------------------------
RUN pip install --no-cache-dir freesasa==2.2.1

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY freesasa/validate_config.sh /opt/tool/validate_config.sh
COPY freesasa/run.sh             /opt/tool/run.sh
COPY freesasa/run_freesasa.py    /opt/tool/run_freesasa.py
COPY freesasa/standardize.sh     /opt/tool/standardize.sh
COPY freesasa/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
