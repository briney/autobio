# autobio-rosetta-base — Rosetta binaries + scoring database
#
# Base image for all Rosetta-derived tool containers. NOT run directly —
# derived images (rosetta-score, rosetta-relax, etc.) add protocol-specific
# scripts and set the autobio entrypoint.
#
# Extends the official rosettacommons/rosetta image, which includes
# pre-compiled binaries and the PyRosetta database. We add the autobio
# protocol dependencies (jq, bc) and the shared score file parser.
#
# Build context: containers/
#   docker build -f containers/rosetta-base/Dockerfile containers/ -t autobio-rosetta-base:3.15

FROM rosettacommons/rosetta:latest

# --- Autobio protocol dependencies (jq, bc required by base-entrypoint.sh) ---
RUN apt-get update && apt-get install -y --no-install-recommends jq bc \
    && rm -rf /var/lib/apt/lists/*

# Rosetta binaries are at /usr/local/bin/ (already on PATH)
# Database is at /usr/local/lib/python3.8/dist-packages/pyrosetta/database/
ENV ROSETTA_DATABASE="/usr/local/lib/python3.8/dist-packages/pyrosetta/database"

# Shared score file parser (used by all derived images' standardize.py)
COPY rosetta-base/parse_scorefile.py /opt/rosetta/parse_scorefile.py

# No ENTRYPOINT — this image is a base for derived tool images
