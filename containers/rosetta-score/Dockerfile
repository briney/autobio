# autobio-rosetta-score — Rosetta structure scoring
#
# Scores a PDB structure using Rosetta's score_jd2 application, producing
# total energy and per-term breakdown in REU (Rosetta Energy Units).
#
# Build context: containers/
#   docker build -f containers/rosetta-score/Dockerfile containers/ -t autobio-rosetta-score:1.0.0
#
# Requires autobio-rosetta-base:3.15 to be built first.

FROM autobio-rosetta-base:3.15

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY rosetta-score/validate_config.sh /opt/tool/validate_config.sh
COPY rosetta-score/run.sh             /opt/tool/run.sh
COPY rosetta-score/standardize.sh     /opt/tool/standardize.sh
COPY rosetta-score/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
