#!/bin/bash
# =============================================================================
# CAMEL Annotation — HTCondor Wrapper Script
# =============================================================================
# Sets up the environment and runs the annotation script.
# Called by HTCondor with: wrapper.sh <chunk_file> <result_file>
#
# Adapted from Prof. Introne's belief extraction wrapper.
#
# Prerequisites:
#   1. Singularity container: vllm-openai.sif (pulled once on login node)
#   2. Conda environment with: pandas, pyarrow, pyyaml, requests
#   3. config2.yaml in the same directory as this script
# =============================================================================

set -euo pipefail

# --- CONFIGURATION (edit these for your setup) ---
CONDA_PATH="${HOME}/miniconda3"          # Path to conda installation
CONDA_ENV="camel-annotation"             # Conda environment name
CONTAINER_PATH="${HOME}/vllm-openai.sif" # Path to Singularity container
CONFIG_FILE="config2.yaml"               # CAMEL config (prompts + labels)

# Model defaults (can be overridden via environment variables)
MODEL="${CAMEL_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
QUANTIZATION="${CAMEL_QUANTIZATION:-awq_marlin}"
TP="${CAMEL_TP:-2}"
THREADS="${CAMEL_THREADS:-16}"
PORT="${CAMEL_PORT:-8900}"

# --- ARGUMENTS ---
CHUNK_FILE="$1"
RESULT_FILE="$2"

echo "============================================================"
echo "CAMEL Annotation — HPC Job"
echo "============================================================"
echo "  Chunk:      ${CHUNK_FILE}"
echo "  Output:     ${RESULT_FILE}"
echo "  Model:      ${MODEL}"
echo "  Quant:      ${QUANTIZATION}"
echo "  TP:         ${TP}"
echo "  Threads:    ${THREADS}"
echo "  Container:  ${CONTAINER_PATH}"
echo "  Config:     ${CONFIG_FILE}"
echo "  Hostname:   $(hostname)"
echo "  Date:       $(date)"
echo "============================================================"

# --- SETUP CONDA ---
echo "Setting up conda environment..."
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# --- VERIFY FILES ---
if [ ! -f "${CHUNK_FILE}" ]; then
    echo "ERROR: Chunk file not found: ${CHUNK_FILE}"
    exit 1
fi

if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Singularity container not found: ${CONTAINER_PATH}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# --- RUN ---
echo "Starting annotation..."
python camel_annotate_hpc.py \
    "${CHUNK_FILE}" \
    "${RESULT_FILE}" \
    --model "${MODEL}" \
    --config "${CONFIG_FILE}" \
    --container "${CONTAINER_PATH}" \
    --quantization "${QUANTIZATION}" \
    --tp "${TP}" \
    --threads "${THREADS}" \
    --port "${PORT}"

EXIT_CODE=$?

echo "============================================================"
echo "Job finished with exit code: ${EXIT_CODE}"
echo "Date: $(date)"
echo "============================================================"

exit ${EXIT_CODE}
