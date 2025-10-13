#!/bin/bash

# This script contains a reusable banner function for display in other scripts.

print_banner() {
    # Color codes for better output
    local CYAN=$'\033[36m'
    local RESET=$'\033[0m'

    cat << EOF

${CYAN}======================================================================${RESET}
${CYAN}                            NeuroCBIR                               ${RESET}
${CYAN}======================================================================${RESET}
${CYAN} A Public Content-Based Image Retrieval System for Whole-Brain and  ${RESET}
${CYAN} Region-Specific MRI Across Multiple Clinical Cohorts.              ${RESET}
${CYAN}                                                                    ${RESET}
${CYAN} Authors: Félix Nieto-del-Amor¹, Jingru Fu¹, J.-Sebastian Muehlboeck², ${RESET}
${CYAN}          Eric Westman², Daniel Ferreira², Rodrigo Moreno¹           ${RESET}
${CYAN}                                                                    ${RESET}
${CYAN} ¹ Division of Biomedical Imaging, KTH Royal Institute of Technology  ${RESET}
${CYAN} ² Division of Clinical Geriatrics, Karolinska Institute              ${RESET}
${CYAN}                                                                    ${RESET}
${CYAN} Contact: fenda@kth.se                                               ${RESET}
${CYAN}======================================================================${RESET}

EOF
}