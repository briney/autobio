# autobio-prodigy — PRODIGY protein-protein binding affinity prediction
#
# Predicts binding affinity (delta-G in kcal/mol, Kd in M) for protein-protein
# complexes using a contact-based linear model. CPU-only, no GPU required.
#
# Build context: containers/
#   docker build -f containers/prodigy/Dockerfile containers/ -t autobio-prodigy:2.4.0

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# --- Autobio protocol dependencies + build tools for freesasa C extension ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq bc build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- PRODIGY + dependencies (Biopython, freesasa, numpy) ---------------------
RUN pip install --no-cache-dir prodigy-prot==2.4.0

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY prodigy/validate_config.sh /opt/tool/validate_config.sh
COPY prodigy/run.sh             /opt/tool/run.sh
COPY prodigy/run_prodigy.py     /opt/tool/run_prodigy.py
COPY prodigy/standardize.sh     /opt/tool/standardize.sh
COPY prodigy/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
