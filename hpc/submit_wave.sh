#!/usr/bin/env bash
# =============================================================================
# submit_wave.sh — submit one CAMEL production wave with run-versioned naming.
# =============================================================================
# Consumes ONE run number for THIS MODEL (via next_run.sh) and supplies the
# naming-schema macros to the .sub, so every job in this wave shares the same
# run<NNNN> and submit date. $(Cluster) + chunk<NNN> keep each file unique
# within the wave.
#
# Run numbers are per-model: llama's first wave and qwen's first wave are BOTH
# run0001. A re-submit of the same model becomes run0002 for that model only.
#
# Usage:
#   submit_wave.sh <model_tag> [chunklist_file]
#     submit_wave.sh qwen2.5-72b                     # default (chunklist_first.txt)
#     submit_wave.sh qwen2.5-72b chunklist_all.txt   # full wave (000..056)
#     submit_wave.sh qwen2.5-72b chunklist_rest.txt  # release 001..056 only
#
# The increment happens exactly once per invocation (one human submit action).
# Invoke from the hpc/ directory — condor_submit resolves the chunklist named in
# `queue chunk_id from $(CHUNKLIST)` relative to the current working directory.
# =============================================================================
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <model_tag> [chunklist_file]" >&2
    exit 2
fi
tag="$1"
chunklist="${2:-}"
sub="${here}/batch_${tag}.sub"

if [ ! -f "$sub" ]; then
    echo "error: submit file not found: $sub" >&2
    exit 1
fi

if [ -n "$chunklist" ] && [ ! -f "$chunklist" ]; then
    echo "error: chunklist not found in \$PWD: $chunklist" >&2
    echo "       run submit_wave.sh from ${here}" >&2
    exit 1
fi

run_no="$("${here}/next_run.sh" "$tag")"   # increments THIS MODEL's counter ONCE
subdate="$(date +%Y%m%d)"

args=(-a "RUN_NO=${run_no}" -a "SUBDATE=${subdate}")
[ -n "$chunklist" ] && args+=(-a "CHUNKLIST = ${chunklist}")

echo "Submitting ${tag}: run${run_no}, date ${subdate}, chunklist ${chunklist:-chunklist_first.txt}"
condor_submit "$sub" "${args[@]}"
