# autobio-rosetta-minimize — Rosetta energy minimization
#
# Minimizes a PDB structure using gradient-based optimization.
# Faster than relax — resolves clashes without full repacking.
#
# Build context: containers/
#   docker build -f containers/rosetta-minimize/Dockerfile containers/ -t autobio-rosetta-minimize:1.0.0
#
# Requires autobio-rosetta-base:3.15 to be built first.

FROM autobio-rosetta-base:3.15

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY rosetta-minimize/validate_config.sh /opt/tool/validate_config.sh
COPY rosetta-minimize/run.sh             /opt/tool/run.sh
COPY rosetta-minimize/standardize.sh     /opt/tool/standardize.sh
COPY rosetta-minimize/standardize.py     /opt/tool/standardize.py
COPY rosetta-minimize/xml/               /opt/tool/xml/
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
