---
description: Detect Apophis↔OrangeGrid divergence in hpc/ code and the known config invariants
---

Check for drift between the Apophis and OrangeGrid copies of the CAMEL `hpc/` code, and verify
the hard-won invariants that keep breaking (CLAUDE.md "Git & Sync Topology"). OG is authoritative
for `hpc/` when in doubt; never blind-copy either direction — always diff first.

Run these checks and report a findings list (file · line · what · which side):

1. **Working-tree vs committed (Apophis):** `git -C /home/szkhan/code_space3 status --short hpc/`
   and `git diff --stat hpc/`. List uncommitted `hpc/` changes — these are the ones at risk of
   being lost to OG drift.
2. **Invariant — V1 engine:** confirm `--kv-cache-dtype fp8_e4m3` is still COMMENTED OUT in
   `hpc/camel_annotate_hpc.py`. An active (uncommented) occurrence means the slow V0 engine is
   back. `grep -n fp8_e4m3 hpc/camel_annotate_hpc.py`.
3. **Invariant — DeepSeek overrides:** show the DeepSeek `MODEL_OVERRIDES` block
   (`max_tokens`, `repetition_penalty`) so I can confirm it matches the value under active tuning.
4. **Invariant — YAML apostrophes:** grep `hpc/config3.yaml` for unescaped apostrophes inside
   single-quoted strings (a lone `'s` that isn't doubled `''s`), especially around the Hate and
   AnalyticalThinking constructs.
5. **OG comparison (if reachable):** if Apophis can reach OG non-interactively, diff the key
   files: `ssh its-og-login5.syr.edu 'cat ~/code_space3/hpc/camel_annotate_hpc.py'` vs local, and
   the wrapper/submit files. If not reachable, say so — do not assume they match.

End with: `Drift: <clean | N divergences>` and, for each divergence, the one-line promote/keep
recommendation. Read-only — do not edit or sync anything without my confirmation.
