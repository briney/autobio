# autobio-rosetta-relax — Rosetta FastRelax structure refinement
#
# Relaxes a PDB structure using cycles of minimization + side-chain repacking.
# Produces refined structures and their Rosetta energy scores.
#
# Build context: containers/
#   docker build -f containers/rosetta-relax/Dockerfile containers/ -t autobio-rosetta-relax:1.0.0
#
# Requires autobio-rosetta-base:3.15 to be built first.

FROM autobio-rosetta-base:3.15

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY rosetta-relax/validate_config.sh /opt/tool/validate_config.sh
COPY rosetta-relax/run.sh             /opt/tool/run.sh
COPY rosetta-relax/standardize.sh     /opt/tool/standardize.sh
COPY rosetta-relax/standardize.py     /opt/tool/standardize.py
COPY rosetta-relax/xml/               /opt/tool/xml/
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
