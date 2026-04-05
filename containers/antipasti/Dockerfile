# autobio-antipasti — ANTIPASTI antibody binding affinity prediction
#
# Predicts antibody-antigen binding affinity (log10 Kd) from PDB complex
# structures using Normal Mode Correlation Maps (DCCM) and a lightweight CNN.
# Requires R + bio3d for Normal Mode Analysis preprocessing.
#
# Build context: containers/
#   docker build -f containers/antipasti/Dockerfile containers/ -t autobio-antipasti:1.0.0

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# --- Autobio protocol dependencies + R + build tools -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq bc git wget \
        r-base r-base-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev libicu-dev \
        libpng-dev make g++ gfortran \
    && rm -rf /var/lib/apt/lists/*

# --- R packages for Normal Mode Analysis ------------------------------------
# bio3d (NMA/DCCM), reticulate (numpy interop), stringr (string utils)
# Install with dependencies to ensure all transitive deps are resolved.
RUN Rscript -e 'install.packages(c("bio3d", "reticulate", "stringr"), repos="https://cloud.r-project.org", dependencies=TRUE)'

# --- ANTIPASTI source, data, and model weights ------------------------------
# Clone at a pinned tag for reproducibility. Weights are in checkpoints/
# and reference data for inference alignment is in data/.
RUN git clone https://github.com/kevinmicha/ANTIPASTI.git /app/antipasti \
    && cd /app/antipasti \
    && git checkout 1.2

# --- Python dependencies ----------------------------------------------------
RUN pip install --no-cache-dir \
        biopython scipy pandas numpy scikit-learn \
        adabelief-pytorch torchmetrics \
    && cd /app/antipasti && pip install --no-cache-dir -e .

# --- Shared entrypoint -------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY antipasti/validate_config.sh /opt/tool/validate_config.sh
COPY antipasti/run.sh             /opt/tool/run.sh
COPY antipasti/inference.py       /opt/tool/inference.py
COPY antipasti/standardize.sh     /opt/tool/standardize.sh
COPY antipasti/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
