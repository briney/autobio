# autobio-stabddg — StaB-ddG binding ddG prediction
#
# ML-based prediction of binding stability changes (ddG) from mutations
# in protein-protein complexes. Uses ProteinMPNN architecture fine-tuned
# on stability and SKEMPI binding data.
#
# Build context: containers/
#   docker build -f containers/stabddg/Dockerfile containers/ -t autobio-stabddg:1.0.0

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# --- Autobio protocol dependencies ------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends jq bc git \
    && rm -rf /var/lib/apt/lists/*

# --- StaB-ddG source and weights --------------------------------------------
# Clone at a pinned commit for reproducibility. Model weights are included
# in the repository under model_ckpts/.
RUN git clone https://github.com/LDeng0205/StaB-ddG.git /app/stabddg \
    && cd /app/stabddg \
    && git checkout 8a8109648448ea0994a219d1d9fd8a59ebfc82a7

# --- Python dependencies ----------------------------------------------------
RUN pip install --no-cache-dir biopython scipy pandas tqdm \
    && pip install --no-cache-dir -e /app/stabddg

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY stabddg/validate_config.sh /opt/tool/validate_config.sh
COPY stabddg/run.sh             /opt/tool/run.sh
COPY stabddg/standardize.sh     /opt/tool/standardize.sh
COPY stabddg/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
