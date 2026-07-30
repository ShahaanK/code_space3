#!/usr/bin/env bash
# =============================================================================
# next_run.sh — atomically consume the next PER-MODEL run number.
# =============================================================================
# Prints the run number THIS wave should use, zero-padded to 4 digits (e.g. 0001),
# then writes back the incremented value. One counter PER MODEL TAG, so every
# model's first production wave is run0001 and a re-submit of that same model
# becomes run0002 (run_counter_<tag>.txt holds the NEXT number to assign).
#
# RUN_NO is the attempt number for that model, NOT the chunk index — all 57
# chunks of a wave share one run number and are distinguished by chunk<NNN>.
# That is what lets select_and_merge.py pick the latest complete attempt.
#
# flock serializes concurrent submits so two waves can never grab the same
# number. Call this ONCE per human submit action (submit_wave.sh does), NEVER
# from inside a .sub (a .sub re-renders per queue item and would bump per-job).
#
# NOTE: the authoritative live counters live on the submit machine (OrangeGrid).
# Any git-tracked run_counter_*.txt is only a seed; do not reseed OG's copies
# from them. The legacy project-global run_counter.txt is retired and unused.
#
# Usage: next_run.sh <model_tag>
# =============================================================================
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <model_tag>" >&2
    exit 2
fi
tag="$1"

counter="${here}/run_counter_${tag}.txt"
exec 9>"${counter}.lock"        # separate lock file; never truncates the counter
flock 9
cur="$(cat "$counter" 2>/dev/null || echo 1)"
printf '%d\n' "$((cur + 1))" > "$counter"   # write-back the next value
printf '%04d\n' "$cur"                        # emit the value THIS wave uses
