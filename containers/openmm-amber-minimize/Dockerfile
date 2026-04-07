# autobio-openmm-amber-minimize — Amber force field energy minimization
#
# Minimizes a protein structure using OpenMM with iterative
# violation-checking (AlphaFold-style). Resolves steric clashes,
# refines geometry, and reports energy in kJ/mol.
#
# Build context: containers/
#   docker build -f containers/openmm-amber-minimize/Dockerfile containers/ -t autobio-openmm-amber-minimize:1.1.0
#
# Requires autobio-openmm-base:8.5.0 to be built first.

FROM autobio-openmm-base:8.5.0

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY openmm-amber-minimize/validate_config.sh /opt/tool/validate_config.sh
COPY openmm-amber-minimize/run.sh             /opt/tool/run.sh
COPY openmm-amber-minimize/run.py             /opt/tool/run.py
COPY openmm-amber-minimize/standardize.sh     /opt/tool/standardize.sh
COPY openmm-amber-minimize/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
