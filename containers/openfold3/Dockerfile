# autobio-openfold3 — OpenFold3 structure prediction (AF3 reproduction)
#
# Builds on the NVIDIA PyTorch 25.02 container with Blackwell (sm_120) GPU
# support.  Installs OpenFold3 from source with pre-cached model weights
# and the autobio three-phase container protocol.
#
# Build context: containers/
#   docker build -f containers/openfold3/Dockerfile containers/ -t autobio-openfold3:1.0.0

FROM nvcr.io/nvidia/pytorch:25.02-py3

# --- System dependencies -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq bc hmmer libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# --- Clone and install OpenFold3 ---------------------------------------------
# Following the Blackwell build instructions from the openfold-3 repository.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/aqlaboratory/openfold-3.git

# Install Python dependencies (PyTorch already provided by base image)
WORKDIR /opt/openfold-3
RUN pip install --no-cache-dir \
    biopython \
    biotite \
    boto3 \
    awscli \
    awscrt \
    cuda-python'<12.9.1' \
    deepspeed \
    func_timeout \
    kalign-python \
    lmdb \
    ml-collections \
    modelcif \
    nvidia-cutlass'<4' \
    pdbeccdutils \
    pytorch-lightning'>=2.1' \
    rdkit \
    wandb \
    pyyaml \
    requests \
    tqdm \
    memory_profiler

# Clone CUTLASS for DeepSpeed evoformer attention kernel
RUN git clone --depth 1 --branch v3.6.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass
ENV CUTLASS_PATH=/opt/cutlass

# Install openfold3 in editable mode (no deps — already installed above)
RUN pip install --no-cache-dir --no-deps -e .

# Set Blackwell arch for torch extensions
ENV TORCH_CUDA_ARCH_LIST="12.0"

# Ensure CUTLASS_PATH and arch list are set before any openfold imports
RUN python3 -c "\
import site, pathlib; \
sp = pathlib.Path(site.getsitepackages()[0]) / 'sitecustomize.py'; \
sp.write_text( \
    'import os\\n' \
    'os.environ.setdefault(\"CUTLASS_PATH\", \"/opt/cutlass\")\\n' \
    'os.environ.setdefault(\"TORCH_CUDA_ARCH_LIST\", \"12.0\")\\n' \
)"

# --- Download model weights at build time (self-contained image) -------------
ENV OPENFOLD_CACHE=/app/openfold3/weights
RUN mkdir -p /app/openfold3/weights && \
    aws s3 cp s3://openfold3-data/openfold3-parameters/of3-p2-155k.pt \
        /app/openfold3/weights/of3-p2-155k.pt --no-sign-request && \
    echo "Downloaded checkpoint ($(du -h /app/openfold3/weights/of3-p2-155k.pt | cut -f1))"

# Write ckpt_root so OpenFold3 finds the pre-cached checkpoint
RUN echo "/app/openfold3/weights" > /app/openfold3/weights/ckpt_root

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY openfold3/validate_config.sh /opt/tool/validate_config.sh
COPY openfold3/run.sh             /opt/tool/run.sh
COPY openfold3/run_openfold3.py   /opt/tool/run_openfold3.py
COPY openfold3/standardize.sh     /opt/tool/standardize.sh
COPY openfold3/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

WORKDIR /workspace
ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
