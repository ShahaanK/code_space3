---
description: Check the OrangeGrid production queue and recent job logs, summarize health
argument-hint: "[jobID] (optional — focus on one cluster)"
---

You are monitoring the CAMEL production wave on OrangeGrid (HTCondor). Goal: a tight status
line I can watch on a loop, and an early warning on the two silent failure modes.

Do this, and keep the output to a compact summary (not a log dump):

1. Query the queue. If Apophis can reach the OG login node non-interactively, run:
   `ssh its-og-login5.syr.edu 'condor_q -nobatch; echo ---; condor_q -run'`
   (add `$ARGUMENTS` as a cluster/jobID filter if given). If SSH is not reachable from here,
   say so and ask me to paste the output via `! ssh its-og-login5.syr.edu condor_q` — do NOT
   fabricate queue state.
2. Flag anything HELD. For held jobs, recall CLAUDE.md: prefer hold-edit-release over
   rm-resubmit. Show the hold reason (`condor_q -held`).
3. Scan the newest `hpc/logs/*.out` and `*.log` for: CUDA/OOM errors, vLLM startup failures,
   V0-vs-V1 engine line (confirm V1 is active — V0 means the fp8 flag crept back), and the
   per-chunk completion cadence.
4. **-1 sentinel:** if any results Feather/CSV has been written, spot-check the -1
   (unparseable) rate — this is how DeepSeek degeneration shows up silently. Call out any model
   above a few percent. (Use `/neg1-check <file>` for a full pass.)

End with one line: `OG: <N running / M idle / K held> · <models active> · <newest chunk done> · <any red flags>`.

Meant to be run via `/loop 10m /og-status`. Respect all CLAUDE.md guardrails: do not submit,
rm, or resubmit anything without my say-so — this is read-only monitoring.
