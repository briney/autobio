# autobio-openmm-amber-relax — Amber force field full relaxation protocol
#
# Performs a multi-phase relaxation: energy minimization, heating ramp,
# NVT/NPT equilibration, and short production dynamics. Supports both
# implicit and explicit solvent. Reports energy in kJ/mol.
#
# Build context: containers/
#   docker build -f containers/openmm-amber-relax/Dockerfile containers/ -t autobio-openmm-amber-relax:1.1.0
#
# Requires autobio-openmm-base:8.5.0 to be built first.

FROM autobio-openmm-base:8.5.0

# --- Shared entrypoint --------------------------------------------------------
COPY base-entrypoint.sh /opt/autobio/base-entrypoint.sh
RUN chmod +x /opt/autobio/base-entrypoint.sh

# --- Tool-specific hook scripts -----------------------------------------------
COPY openmm-amber-relax/validate_config.sh /opt/tool/validate_config.sh
COPY openmm-amber-relax/run.sh             /opt/tool/run.sh
COPY openmm-amber-relax/run.py             /opt/tool/run.py
COPY openmm-amber-relax/standardize.sh     /opt/tool/standardize.sh
COPY openmm-amber-relax/standardize.py     /opt/tool/standardize.py
RUN chmod +x /opt/tool/*.sh

ENTRYPOINT ["/opt/autobio/base-entrypoint.sh"]
