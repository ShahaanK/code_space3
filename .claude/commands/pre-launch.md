---
description: Production readiness checklist before launching a CAMEL 56K wave on OrangeGrid
argument-hint: "[model name] (optional)"
---

Gate a CAMEL production wave before it burns A100 time. Walk this checklist for
`$ARGUMENTS` (or all enabled models if none given) and report PASS/FAIL/UNKNOWN per item with
evidence. Do NOT submit anything — this is a go/no-go check I run before `condor_submit`.

**Blockers (any FAIL = do not launch):**
1. **DeepSeek gate:** is the model DeepSeek-R1? If so, is the -1 degeneration resolved? Per
   CLAUDE.md it is an OPEN blocker — DeepSeek must NOT go to the full corpus until decided. FAIL
   for DeepSeek unless I explicitly override.
2. **F1 regression:** has the config2-vs-config3 F1 regression been isolated (run `/f1-diff`)?
   Launching on a degraded config wastes 8+ days. FAIL if unresolved for the config being used.
3. **Chunks exist:** `hpc/chunks/chunk_manifest.json` present with the expected `total_chunks`
   (~57 at 1000 texts/chunk) and required fields (total_chunks, chunk_size, total_unprocessed,
   chunks[] with chunk_id/filename/rows/original_start_index/original_end_index).

**Config sanity:**
4. Correct cluster config for OG: TP=1, `+request_gpus=1`, A100 80GB targeting with
   `TARGET.GPUs >= 2`, `should_transfer_files=NO` (and NO `when_to_transfer_output`), explicit
   HOME in the environment ClassAd, and `CAMEL_MODEL` set to the full HF path.
5. V1 engine invariant: `fp8_e4m3` still commented out in `hpc/camel_annotate_hpc.py`.
6. Column convention: the corpus Feather uses `Number`/`text` and `CAMEL_CONFIG=config3.yaml`.
7. Per-model wrapper exists and bakes the right `--model` (`generate_batch_jobs.py` output).

**Process:**
8. Introne subset protocol: plan is submit chunk 1 for the model, verify (queue + `/neg1-check`
   on the first result), THEN release the rest — not all 57 at once.

End with a single verdict: `LAUNCH OK` or `DO NOT LAUNCH — <blocking items>`.
