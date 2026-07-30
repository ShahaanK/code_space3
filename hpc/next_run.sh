#!/usr/bin/env bash
# =============================================================================
# next_run.sh — atomically consume the next GLOBAL project run number.
# =============================================================================
# Prints the run number THIS wave should use, zero-padded to 4 digits (e.g. 0001),
# then writes back the incremented value. One project-global counter for ALL
# models and chunks (run_counter.txt holds the NEXT number to assign).
#
# flock serializes concurrent submits so two waves can never grab the same
# number. Call this ONCE per human submit action (submit_wave.sh does), NEVER
# from inside a .sub (a .sub re-renders per queue item and would bump per-job).
#
# NOTE: the authoritative live counter lives on the submit machine (OrangeGrid).
# The git-tracked run_counter.txt is only a seed; do not reseed OG's copy from it.
# =============================================================================
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
counter="${here}/run_counter.txt"
exec 9>"${counter}.lock"        # separate lock file; never truncates the counter
flock 9
cur="$(cat "$counter" 2>/dev/null || echo 1)"
printf '%d\n' "$((cur + 1))" > "$counter"   # write-back the next value
printf '%04d\n' "$cur"                        # emit the value THIS wave uses
