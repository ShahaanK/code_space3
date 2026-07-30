#!/bin/bash
# =============================================================================
# CAMEL neg1_watch.sh — Apophis-side -1 (unparseable) rate sentinel
# =============================================================================
# Runs ON APOPHIS, never on OG -- per project convention, result analysis
# (pandas/eval) always happens on Apophis, where the gold key and myenv live.
# This script polls OG via SSH for newly-completed (exit 0) jobs, pulls each
# finished chunk's result file back to Apophis, computes the -1 (unparseable)
# rate locally with pandas, and posts a Slack alert ONLY if the rate exceeds
# THRESHOLD_PCT. Silent otherwise -- this is a "something's wrong" sentinel,
# not a per-chunk status feed (use /og-status or /neg1-check manually for
# routine checks).
#
# Complements condor_alert.sh (runs on OG, watches HTCondor job STATE only --
# HELD/FAILED/auto-released/digest). This script watches annotation QUALITY,
# which can only be checked by reading the actual result data.
#
# Usage: nohup ./neg1_watch.sh > neg1_watch.out 2>&1 &
# Stop:  pkill -f neg1_watch.sh
# =============================================================================

set -uo pipefail

OG_HOST="its-og-login5.syr.edu"
OG_HPC_DIR="code_space3/hpc"
# Slack webhook is NOT stored in this file. It lives in ~/.camel_secrets.env
# (chmod 600, outside the git repo so no .gitignore mistake can publish it).
# Override the location with CAMEL_SECRETS if needed.
CAMEL_SECRETS="${CAMEL_SECRETS:-$HOME/.camel_secrets.env}"
# shellcheck source=/dev/null
[ -r "$CAMEL_SECRETS" ] && . "$CAMEL_SECRETS"
WEBHOOK_URL="${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL unset -- create $CAMEL_SECRETS with: export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...}"
POLL_INTERVAL=1800          # 30 min -- result files only land once per
                            # multi-hour chunk job finishes; no need to poll fast
THRESHOLD_PCT=10            # matches CLAUDE.md's documented degeneration threshold
STATE_FILE="/home/szkhan/code_space3/hpc/.neg1_watch_processed.txt"
INCOMING_DIR="/home/szkhan/code_space3/hpc/neg1_watch_incoming"
PYBIN="/home/szkhan/myenv/bin/python3"

mkdir -p "$INCOMING_DIR"
touch "$STATE_FILE"

notify() {
    local text="$1"
    local payload
    payload=$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$text")
    curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$WEBHOOK_URL" > /dev/null
}

echo "neg1_watch.sh started on Apophis. Poll every ${POLL_INTERVAL}s. Alert threshold: ${THRESHOLD_PCT}%."
notify "🟢 neg1_watch.sh started on Apophis (poll every $((POLL_INTERVAL/60))min). Alerts only if a completed chunk's -1 rate exceeds ${THRESHOLD_PCT}%."

while true; do
    # ClusterId ProcId ExitCode Args, one line per job in OG history.
    # Args (per wrapper.sh's `arguments = $(chunk_file) $(result_file)`) is
    # "<chunk_file> <result_file> [extra flags]" -- 2nd field is result_file.
    completed=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$OG_HOST" \
        "cd $OG_HPC_DIR && condor_history \$USER -af ClusterId ProcId ExitCode Args" 2>/dev/null \
        | awk '$3 == 0')

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        cid=$(echo "$line" | awk '{print $1}')
        pid=$(echo "$line" | awk '{print $2}')
        job="${cid}.${pid}"

        if grep -qx "$job" "$STATE_FILE"; then
            continue
        fi

        result_file=$(echo "$line" | awk '{print $5}')

        if [ -z "$result_file" ]; then
            echo "$job" >> "$STATE_FILE"   # nothing parseable, don't retry forever
            continue
        fi

        local_path="${INCOMING_DIR}/${job}_$(basename "$result_file")"

        if scp -o ConnectTimeout=15 "${OG_HOST}:${OG_HPC_DIR}/${result_file}" "$local_path" 2>/dev/null; then
            result=$("$PYBIN" - "$local_path" <<'PYEOF'
import sys
import pandas as pd

path = sys.argv[1]
try:
    df = pd.read_feather(path)
except Exception as e:
    print(f"ERROR|{e}")
    sys.exit(0)

meta = {"text_id", "prompt_id", "prompt_name", "model", "provider",
        "temperature", "run_number", "original_index"}
lbl_cols = [c for c in df.columns if not c.startswith("response__") and c not in meta]
total = sum(len(df[c]) for c in lbl_cols)
neg1 = sum(int((df[c] == -1).sum()) for c in lbl_cols)
rate = 100 * neg1 / total if total else 0.0
model = df["model"].iloc[0] if "model" in df.columns and len(df) else "unknown"
print(f"OK|{rate:.2f}|{len(df)}|{model}")
PYEOF
)
            if [[ "$result" == ERROR* ]]; then
                echo "  WARNING: could not read $local_path (${result#ERROR|})"
            else
                pct=$(echo "$result" | cut -d'|' -f2)
                rows=$(echo "$result" | cut -d'|' -f3)
                model=$(echo "$result" | cut -d'|' -f4)
                above=$("$PYBIN" -c "print(1 if float('$pct') > $THRESHOLD_PCT else 0)")
                if [ "$above" = "1" ]; then
                    notify "🚨 Chunk quality alert: job $job ($model, $rows rows) has a ${pct}% -1 (unparseable) rate -- above the ${THRESHOLD_PCT}% threshold. File: $result_file"
                fi
            fi
        else
            echo "  WARNING: scp failed for $job ($result_file)"
        fi

        echo "$job" >> "$STATE_FILE"
    done <<< "$completed"

    sleep "$POLL_INTERVAL"
done
