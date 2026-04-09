# autobio-protenix — Protenix-v2 structure prediction
#
# Builds on the NVIDIA PyTorch 25.02 container.  Installs Protenix from
# source with pre-cached model weights and the autobio three-phase
# container protocol.  Includes a custom MSA fetch script that queries
# a ColabFold server and converts results to Protenix's paired/unpaired
# per-chain MSA format.
#
# Build context: containers/
#   docker build -f containers/protenix/Dockerfile containers/ -t autobio-protenix:1.0.0

FROM nvcr.io/nvidia/pytorch:25.02-py3

# --- System dependencies -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq bc wget hmmer kalign \
    && rm -rf /var/lib/apt/lists/*

# --- Clone CUTLASS v3.5.1 (required by Protenix DeepSpeed kernels) -----------
RUN git clone --depth 1 --branch v3.5.1 \
    https://github.com/NVIDIA/cutlass.git /opt/cutlass
ENV CUTLASS_PATH=/opt/cutlass

# --- Clone and install Protenix -----------------------------------------------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/bytedance/Protenix.git protenix

WORKDIR /opt/protenix

# Install Python dependencies (PyTorch already provided by base image)
RUN pip install --no-cache-dir \
    biopython \
    biotite \
    rdkit \
    deepspeed \
    triton \
    fair-esm \
    modelcif \
    ml-collections \
    pydantic \
    scipy \
    scikit-learn \
    cuequivariance-ops-torch-cu12 \
    cuequivariance-torch \
    wandb \
    click

# Install protenix in editable mode (no deps — already installed above)
RUN pip install --no-cache-dir --no-deps -e .

# --- Pre-cache model weights and data at build time (self-contained image) ----
# Protenix resolves its cache root via PROTENIX_ROOT_DIR (default: $HOME).
# Checkpoints go to $PROTENIX_ROOT_DIR/checkpoint/, data cache to
# $PROTENIX_ROOT_DIR/common/.  Pre-caching avoids runtime downloads that may
# fail if the upstream TOS server restricts access.
ENV PROTENIX_ROOT_DIR=/root
RUN python3 <<'PYEOF'
import os, urllib.request
from protenix.web_service.dependency_url import URL

root = os.environ.get("PROTENIX_ROOT_DIR", "/root")

def dl(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")

# Checkpoint
dl(URL["protenix_base_default_v1.0.0"], f"{root}/checkpoint/protenix_base_default_v1.0.0.pt")

# Data cache
for key in ("ccd_components_file", "ccd_components_rdkit_mol_file",
            "pdb_cluster_file", "obsolete_release_data_csv"):
    url = URL[key]
    dl(url, f"{root}/common/{os.path.basename(url)}")

print("Pre-cache complete.")
PYEOF

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY protenix/validate_config.sh /opt/tool/validate_config.sh
COPY protenix/run.sh             /opt/tool/run.sh
COPY protenix/fetch_msa.py       /opt/tool/fetch_msa.py
COPY protenix/standardize.sh     /opt/tool/standardize.sh
COPY protenix/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

WORKDIR /workspace
ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
