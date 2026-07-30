#!/usr/bin/env bash
# =============================================================================
# submit_wave.sh — submit one CAMEL production wave with run-versioned naming.
# =============================================================================
# Consumes ONE global run number (via next_run.sh) and supplies the naming-schema
# macros to the .sub, so every job in this wave shares the same run<NNNN> and
# submit date. $(Cluster) + chunk<NNN> keep each file unique within the wave.
#
# Usage:
#   submit_wave.sh <model_tag> [chunklist_file]
#     submit_wave.sh qwen2.5-72b                      # first wave (chunklist_first.txt)
#     submit_wave.sh qwen2.5-72b chunklist_rest.txt   # release the rest (001..056)
#
# The increment happens exactly once per invocation (one human submit action).
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

run_no="$("${here}/next_run.sh")"     # increments the global counter ONCE, here
subdate="$(date +%Y%m%d)"

args=(-a "RUN_NO=${run_no}" -a "SUBDATE=${subdate}")
[ -n "$chunklist" ] && args+=(-a "CHUNKLIST = ${chunklist}")

echo "Submitting ${tag}: run${run_no}, date ${subdate}, chunklist ${chunklist:-chunklist_first.txt}"
condor_submit "$sub" "${args[@]}"
