---
description: Report the -1 (unparseable) rate of a CAMEL result file — the silent degeneration sentinel
argument-hint: "<path to result .feather or .csv>"
---

Check a CAMEL annotation result file for the `-1` (UNCLEAR/ERROR) rate. This is the silent
failure signal: `parse_yes_no()` returns -1 on degenerate output, so a bad run looks fine until
you measure this. CLAUDE.md reference point — the DeepSeek 100-text run hit **36.5% -1**.

Target file: `$ARGUMENTS` (if empty, ask me which result file, or list recent ones in
`outputs/` and `test_results_*_full.csv`).

Steps:
1. Activate the venv: `source ~/myenv/bin/activate`. Read-only — never modify the file.
2. Load it (pandas for CSV; `fcat` or pandas/pyarrow for Feather). The 25 label columns hold
   1/0/-1 (CLAUDE.md "Column Conventions"). Identify them (exclude `response__*` and metadata).
3. Compute:
   - overall -1 rate across all label cells,
   - per-label -1 rate, sorted worst-first,
   - if a `model` column exists, break the -1 rate down per model.
4. Verdict:
   - **< ~2%** → clean, fine to proceed.
   - **2–10%** → suspicious, inspect a few `response__<label>` samples for the worst label.
   - **> 10%** → degeneration; for DeepSeek this is the known blocker — recommend `/deepseek-triage`
     and do NOT treat the run as usable.
5. If the rate is high, pull 2–3 verbatim `response__<label>` snippets so I can see the failure mode.

Output a compact table + one-line verdict per model.
