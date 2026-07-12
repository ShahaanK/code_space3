#!/bin/bash

LOG_FILE="/home/szkhan/code_space3/hpc/logs/test_og.log"

WEBHOOK_URl="https://hooks.slack.com/services/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"
#WEBHOOK_URL="curl -X POST -H 'Content-type: application/json' --data '{"text":"Hello, World!"}' https://hooks.slack.com/services/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"   # your real webhook

echo "Monitoring: $LOG_FILE"

while true; do

    if [ ! -f "$LOG_FILE" ]; then

        sleep 30; continue

    fi

    # 005 = terminated (may be success OR nonzero exit), 009 = aborted, 012 = held

    if grep -qE "^005 " "$LOG_FILE"; then

        # capture exit status if present

        STATUS=$(grep -A2 "^005 " "$LOG_FILE" | grep -oE "return value [0-9]+" | tail -1)

        curl -s -X POST -H 'Content-type: application/json' \

             --data "{\"text\":\"✅ CAMEL DeepSeek job terminated. $STATUS\"}" "$WEBHOOK_URL"

        break

    elif grep -qE "^012 " "$LOG_FILE"; then

        REASON=$(grep -A1 "^012 " "$LOG_FILE" | tail -n1 | sed 's/^[ \t]*//')

        curl -s -X POST -H 'Content-type: application/json' \

             --data "{\"text\":\"⚠️ CAMEL DeepSeek job HELD. Reason: $REASON\"}" "$WEBHOOK_URL"

        break

    elif grep -qE "^009 " "$LOG_FILE"; then

        curl -s -X POST -H 'Content-type: application/json' \

             --data '{"text":"❌ CAMEL DeepSeek job aborted."}' "$WEBHOOK_URL"

        break

    fi

    sleep 30

done
