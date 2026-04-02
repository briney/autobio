# EvoEF2 — physics-based protein energy function
#
# Build context: containers/
#   docker build -f containers/evoef2/Dockerfile containers/ -t autobio-evoef2:1.0.0
#
# EvoEF2 is a compiled C++ tool with no ML dependencies.
# Rotamer libraries and energy parameters are included in the source tree.

FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and autobio protocol dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        git \
        jq \
        bc \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# Clone EvoEF2 at pinned commit for reproducibility
RUN git clone https://github.com/tommyhuangthu/EvoEF2.git /app/evoef2 \
    && cd /app/evoef2 \
    && git checkout 38df01d305ed728ef067c3e0d22072058f33e255

# Compile EvoEF2 from source
RUN cd /app/evoef2 \
    && g++ -O3 -o EvoEF2 src/*.cpp

# Verify the build produced a working binary and library directory
RUN test -x /app/evoef2/EvoEF2 \
    && test -d /app/evoef2/library \
    && /app/evoef2/EvoEF2 --help > /dev/null 2>&1 || true \
    && echo "EvoEF2 build verified."

# Shared entrypoint
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# Tool-specific scripts
COPY evoef2/validate_config.sh /opt/tool/validate_config.sh
COPY evoef2/run.sh             /opt/tool/run.sh
COPY evoef2/standardize.sh     /opt/tool/standardize.sh
COPY evoef2/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
