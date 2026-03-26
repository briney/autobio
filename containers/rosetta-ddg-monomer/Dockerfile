# autobio-rosetta-ddg-monomer — Rosetta DDG stability prediction
#
# Predicts change in protein stability (DDG) upon point mutation using
# Rosetta's high-resolution DDG protocol.
#
# Build context: containers/
#   docker build -f containers/rosetta-ddg-monomer/Dockerfile containers/ -t autobio-rosetta-ddg-monomer:1.0.0
#
# Requires autobio-rosetta-base:3.15 to be built first.

FROM autobio-rosetta-base:3.15

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY rosetta-ddg-monomer/validate_config.sh /opt/tool/validate_config.sh
COPY rosetta-ddg-monomer/run.sh             /opt/tool/run.sh
COPY rosetta-ddg-monomer/standardize.sh     /opt/tool/standardize.sh
COPY rosetta-ddg-monomer/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
