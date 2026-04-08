# autobio-openmm-md-simulate — Molecular dynamics simulation with OpenMM
#
# Runs a full MD protocol: structure cleanup, solvation, energy minimization,
# NVT/NPT equilibration, and production MD with trajectory and energy output.
#
# Build context: containers/
#   docker build -f containers/openmm-md-simulate/Dockerfile containers/ -t autobio-openmm-md-simulate:1.1.0
#
# Requires autobio-openmm-base:8.5.0 to be built first.

FROM autobio-openmm-base:8.5.0

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY openmm-md-simulate/validate_config.sh /opt/tool/validate_config.sh
COPY openmm-md-simulate/run.sh             /opt/tool/run.sh
COPY openmm-md-simulate/run.py             /opt/tool/run.py
COPY openmm-md-simulate/standardize.sh     /opt/tool/standardize.sh
COPY openmm-md-simulate/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
