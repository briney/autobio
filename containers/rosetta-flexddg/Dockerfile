# autobio-rosetta-flexddg — Flex-ddG binding DDG prediction
#
# Ensemble-based DDG prediction at protein-protein interfaces using
# backrub conformational sampling. More accurate than ddg_monomer for
# interface mutations.
#
# Build context: containers/
#   docker build -f containers/rosetta-flexddg/Dockerfile containers/ -t autobio-rosetta-flexddg:1.0.0
#
# Requires autobio-rosetta-base:3.15 to be built first.

FROM autobio-rosetta-base:3.15

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY rosetta-flexddg/validate_config.sh /opt/tool/validate_config.sh
COPY rosetta-flexddg/run.sh             /opt/tool/run.sh
COPY rosetta-flexddg/standardize.sh     /opt/tool/standardize.sh
COPY rosetta-flexddg/standardize.py     /opt/tool/standardize.py
COPY rosetta-flexddg/xml/               /opt/tool/xml/
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
