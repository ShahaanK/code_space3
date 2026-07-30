#!/bin/bash
# =============================================================================
# CAMEL condor_alert.sh — Slack watcher for HTCondor job state (runs on OG)
# =============================================================================
# Rewritten 2026-07-30. The previous version had four compounding bugs: a dead
# Slack webhook, a WEBHOOK_URl/WEBHOOK_URL case typo (curl always posted to an
# empty URL), it was not running as a process, and it only ever watched one
# hardcoded log file (test_og.log) for one job's lifecycle, then exited after
# the first terminal event -- it could never have covered the current
# production wave (114 Llama+Qwen chunks) or the DeepSeek diagnostic jobs.
#
# This version polls `condor_q $USER` directly instead of tailing a specific
# log file, so it automatically covers every job you own -- present and
# future -- with no per-job configuration. It runs indefinitely (no `break`).
#
# Alerts on:
#   - a job going HELD (with hold reason)
#   - a held job being auto-released (periodic_release firing), so a retry
#     is visible instead of silently happening
#   - a job disappearing from the queue with a nonzero exit code (failure)
#   - a 6-hour digest (running/idle/held counts) as a heartbeat that the
#     watcher itself is still alive
# Deliberately does NOT alert on every successful completion -- with 100+
# chunks in the production wave, a per-chunk success message would be noise.
# Use /og-status or condor_history for progress.
#
# This script only watches HTCondor job STATE. It does not read result data
# or compute -1 rates -- that analysis runs on Apophis (see neg1_watch.sh),
# never on OG, per project convention.
#
# Usage: nohup ./condor_alert.sh > condor_alert.out 2>&1 &
# Stop:  pkill -f condor_alert.sh
# =============================================================================

set -uo pipefail

# Slack webhook is NOT stored in this file. It lives in ~/.camel_secrets.env
# (chmod 600, outside the git repo so no .gitignore mistake can publish it).
# Override the location with CAMEL_SECRETS if needed.
CAMEL_SECRETS="${CAMEL_SECRETS:-$HOME/.camel_secrets.env}"
# shellcheck source=/dev/null
[ -r "$CAMEL_SECRETS" ] && . "$CAMEL_SECRETS"
WEBHOOK_URL="${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL unset -- create $CAMEL_SECRETS with: export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...}"
POLL_INTERVAL=60
DIGEST_INTERVAL=21600   # 6 hours

notify() {
    # JSON-escape via python3 rather than naive string interpolation, so a
    # hold reason containing a quote or backslash can't break the payload.
    local text="$1"
    local payload
    payload=$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$text")
    curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$WEBHOOK_URL" > /dev/null
}

status_name() {
    case "$1" in
        1) echo "Idle" ;;
        2) echo "Running" ;;
        3) echo "Removed" ;;
        4) echo "Completed" ;;
        5) echo "Held" ;;
        6) echo "Transferring" ;;
        7) echo "Suspended" ;;
        *) echo "Unknown($1)" ;;
    esac
}

echo "Watching condor_q for $USER (poll ${POLL_INTERVAL}s, digest every ${DIGEST_INTERVAL}s)."
notify "🟢 condor_alert.sh watcher started for $USER. Poll: ${POLL_INTERVAL}s. Alerts on HELD, auto-released, and FAILED jobs. Digest every 6h."

declare -A last_status
last_digest=$(date +%s)

while true; do
    declare -A seen_now
    running=0; idle=0; held=0

    while read -r cid pid status; do
        [ -z "${cid:-}" ] && continue
        job="${cid}.${pid}"
        seen_now["$job"]=1

        case "$status" in
            1) idle=$((idle+1)) ;;
            2) running=$((running+1)) ;;
            5) held=$((held+1)) ;;
        esac

        prev="${last_status[$job]:-}"

        if [ "$status" = "5" ] && [ "$prev" != "5" ]; then
            reason=$(condor_q "$job" -af HoldReason 2>/dev/null | head -1)
            notify "⚠️ Job $job HELD. Reason: ${reason:-unknown}"
        elif [ "$prev" = "5" ] && [ "$status" != "5" ]; then
            notify "🔄 Job $job auto-released (now $(status_name "$status")), retrying."
        fi

        last_status["$job"]="$status"
    done < <(condor_q "$USER" -af ClusterId ProcId JobStatus 2>/dev/null)

    # Jobs that vanished from condor_q since last poll: check condor_history
    # for a nonzero exit code (failure) -- silent if exit 0 (success).
    for job in "${!last_status[@]}"; do
        if [ -z "${seen_now[$job]:-}" ]; then
            exitcode=$(condor_history "$job" -af ExitCode 2>/dev/null | head -1)
            if [ -n "$exitcode" ] && [ "$exitcode" != "0" ]; then
                notify "❌ Job $job finished with nonzero exit code $exitcode."
            fi
            unset 'last_status[$job]'
        fi
    done

    now=$(date +%s)
    if [ $((now - last_digest)) -ge "$DIGEST_INTERVAL" ]; then
        total=${#last_status[@]}
        notify "📊 6h digest for $USER: $running running, $idle idle, $held held ($total tracked jobs)."
        last_digest=$now
    fi

    sleep "$POLL_INTERVAL"
done
