# WORKLOG.md — CAMEL Cultural & Moral Bias in LLM Annotation

**Reverse-chronological development narrative.** Latest sessions on top.

---

## 2026-07-06 — Return After Gap: DeepSeek Penalty Fix, 36.5% Failure, V1 Near-Miss, Version-Control Reconciliation

First session after roughly three weeks away. Began by reconstructing state: the committed docs had drifted badly (WORKLOG topped at 5/19 with 5/21, 5/28, and June work living only in side files or unlogged; WORKPLAN M8 checkboxes stale; CLAUDE.md two months old). Root cause is the read-only OG GitHub PAT plus project-knowledge copies not being re-uploaded. Most of the session was spent deploying the DeepSeek repetition-penalty fix end to end, hitting a real failure, and cleaning up the version-control mess that had been causing silent drift all along.

### DeepSeek Repetition-Penalty Fix (code, not config)

Recon before editing established the fix had to go in code, not YAML. On the HPC path, `camel_annotate_hpc.py` reads sampling params from `MODEL_OVERRIDES` constants and the `call_vllm` signature, not from config3's per-model `temperature`/`max_tokens`, so editing config3 would have been a no-op. There is also no `adapter.py` on the HPC path (the earlier task pointed at it wrongly); the runner builds its request payload by hand over urllib, which means `repetition_penalty` goes straight into the payload dict as a top-level key with no `extra_body` wrapper needed.

The edit, applied to `camel_annotate_hpc.py` via Claude Code on Apophis: DeepSeek `max_tokens` 1024 to 2048, a new `DEEPSEEK_REPETITION_PENALTY = 1.15` constant added to the DeepSeek `MODEL_OVERRIDES` entry, and conditional plumbing through `call_vllm` (a `repetition_penalty=None` param injected into the payload only when set) and `annotate_chunk` and `main`, so Llama and Qwen payloads stay byte-identical. temperature stays 0 for all models.

### V1 Near-Miss (caught before it reverted the engine)

The Apophis copy Claude Code edited turned out to still have the fp8 KV-cache flag active (V0 state); the fp8 removal for V1 had only ever been made on OG. Pushing the Apophis file to OG therefore silently re-added fp8 and reverted the V1 engine. Caught it by verifying inside the running job: `condor_ssh_to_job` plus `nvidia-smi` showed 92% GPU util (healthy, genuinely annotating), then a grep showed fp8 re-active. Re-commented the fp8 line on OG with `sed`, re-verified both the penalty hits and the commented fp8 line, then re-synced OG back to Apophis so both boxes finally matched on the true V1-plus-penalty file. Lesson reinforced: diff before promoting a file across boxes, never blind-rsync.

### CAMEL_MODEL Trap and Output-Name Corruption

Two more pre-launch catches. The template `wrapper.sh` defaults `MODEL` to Llama unless `CAMEL_MODEL` is set in the submit environment, and `test_job_og.sub` had no `CAMEL_MODEL`, so the DeepSeek job would have run Llama (and skipped the penalty override, which keys on the exact DeepSeek HF string). Added `CAMEL_MODEL=Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ` to the environment line. Separately, a nano edit had corrupted the `arguments` line into a duplicated `...feather1~arguments...` string; fixed with a line-targeted `sed`. Both caught before burning a run.

### DeepSeek Re-Run Result: 36.5% Unparseable

The re-run completed. A single live curl against the running vLLM server (with the penalty) returned clean coherent output ending in YES, which looked like success. But the full 100-text evaluation on Apophis showed a 36.5% unparseable (-1) rate. The failing outputs are short and token-level incoherent (malformed strings like "whether whether", nested-tag salad), not the long repetition loops of the original failure and not truncation. So `repetition_penalty` 1.15 at temperature 0 traded the original repetition collapse for a different, penalty-induced degeneration on the longer annotation prompts. The OG log confirmed the batch job did apply the penalty (`Max tokens: 2048`, `Repetition penalty: 1.15`), so this is a real result, not a plumbing miss.

This is the second deterministic decoding config to fail DeepSeek at the 100-text gate (bare greedy degenerated via repetition; penalized greedy degenerates via incoherence). M8.T10h reopens. Decision pending: try a gentler `repetition_penalty` (~1.05) as one more cheap 100-text test; if that also fails, escalate to the June diagnosis doc's Option 4 (document DeepSeek as a reasoning-model exception in the methods, or swap the Eastern reasoning slot for a controllable non-thinking model like Qwen3, which changes the roster and is an Introne/Atari decision). Full corpus stays gated for DeepSeek; Llama and Qwen are unaffected.

### Version-Control Reconciliation

Cleaned up the OG-vs-Apophis-vs-GitHub drift that had caused the session's friction. Established via `git status` and diffs that OG was authoritative for `hpc/` code: OG had uncommitted edits to `config3.yaml` (the line 491/492/494 apostrophe fix), `wrapper.sh` (CAMEL_CONFIG override, CUDA_VISIBLE_DEVICES UUID-to-integer conversion, HF offline env), plus the submit files and `condor_alert.sh`, none of which were on Apophis or GitHub. `camel_annotate_hpc.py` was already aligned. Promoted OG's authoritative files to Apophis (scoped rsync, avoiding the phantom `chunks/logs/results` symlink deletions), staged the six real source files (not the `.bak`/`.save`/CSV cruft), and committed from Apophis, whose SSH remote can push (OG's HTTPS PAT is read-only). All three now aligned on `hpc/` code for the first time. The config3 apostrophe bug is finally committed everywhere.

### Documentation and Housekeeping

Refreshed CLAUDE.md (local to Apophis, untracked) with corrected OrangeGrid status, the V0/V1 engine note, the DeepSeek override, the CAMEL_MODEL requirement, a Git and Sync topology section, and DeepSeek recorded as an active problem. Confirmed the Slack watcher (`condor_alert.sh`) works with a real webhook. Regenerated VISION.md (net-new) and this WORKLOG/WORKPLAN update.

### Open Items

DeepSeek decision (1.05 test then Option 4 if needed); F1 regression diagnosis (config2 vs config3 on Llama; still ~27% below March, gates production); Apophis re-baseline (195 RPM calls vs rows); chunk corpus at 1,000/chunk (57 chunks) and launch Llama + Qwen production wave once DeepSeek is settled, first chunk per model verified before releasing the rest; regenerate OG PAT with Read+Write (M8.T17).

---

## 2026-06-13 — DeepSeek Failure Diagnosis Document and NVIDIA Use Case 1 Slide Rebuild

Session split between a faculty-facing diagnosis document for the DeepSeek degeneration and a rebuild of the NVIDIA deck's Use Case 1 slide.

### DeepSeek Failure Diagnosis Document

Built a five-page Word document for Prof. Introne dissecting the DeepSeek-R1 32B degeneration, with verbatim failure examples pulled from the results CSV, a per-construct parse-failure table, a diagnosis, and four ranked remediation options. Every option respects the non-negotiable temperature-0 constraint required for bias-faithful annotation. The dominant fix identified as a deterministic repetition penalty (`repetition_penalty ~1.15` or `frequency_penalty ~0.4`), not raising max_tokens. The degeneration is visible within the first 500 characters of output, so it is a repetition-collapse failure mode and not a truncation artifact. A prior March hard crash from exceeding max_model_len was confirmed to be a distinct earlier failure. (Note: the 7/6 re-run showed the 1.15 penalty itself induces a different degeneration at scale; see that entry.)

### NVIDIA Use Case 1 Slide Rebuild

Rebuilt the Use Case 1 slide around cost economics rather than throughput mechanics after identifying the original framing was analytically wrong. The old slide presented a 50-to-157 tokens/sec figure as a tuning gain when it was actually a cross-machine comparison with the A6000 mislabeled "tuned." The real tuning win was on OrangeGrid itself, 63 to 224 RPM on the same hardware, which then beat the A6000. The corrected slide reframes around cost: the full benchmark is 4.26M calls per model, roughly 20.4B tokens across three models, priced at the cheapest API host totals about $5,135, exceeding the $2,500 budget cap, making on-prem the only viable path. Line items: Llama 3.3 70B and Qwen 2.5 72B at approximately $1,312 each, DeepSeek-R1 32B at approximately $2,512 despite being smallest, because its 1,024-token reasoning output drives roughly 13x the output volume. Headline: "On-prem wasn't the cheap option, it was the only one." Three wrong framings corrected: query-to-model routing changed to accurate model-to-hardware routing language, "FP precision trade-offs to cut compute" removed as backwards (fp8 was the bottleneck), "8B vs 70B" flagged as NVIDIA's canonical example not a project measurement.

---

## 2026-06-05 — V1 Engine Unlock and 100-Text Multi-Model Characterization

The session that overturned the 5/21 conclusion that OrangeGrid was structurally 3x slower than Apophis.

### V1 Engine Unlock

Removed `--kv-cache-dtype fp8_e4m3` from the vLLM invocation in `camel_annotate_hpc.py`, which was forcing vLLM 0.10.1.dev onto the legacy V0 engine. With the flag gone the V1 engine activated, and Llama 3.3 70B throughput jumped from the locked 63 RPM (V0) to 224 RPM on the same single A100, roughly 3.6x, exceeding the Apophis 195 RPM reference. The tradeoff is higher KV cache memory per request, comfortably absorbed by the 80GB card. Removing fp8 is a pure win, not a precision tradeoff. V1 confirmed annotation-neutral: identical macro F1 across all three prompting strategies between V0 and V1.

### 100-Text Multi-Model Characterization

Ran a multi-model characterization on the 100-text test chunk under the V1 engine. Llama 3.3 70B and Qwen 2.5 72B both completed. DeepSeek-R1 32B stalled on cluster contention after two resolved configuration failures and did not finish.

### Central Emerging Finding

Llama over-predicts and Qwen under-predicts, and the two land at near-identical aggregate F1 through opposite failure modes. Both collapse to near-zero F1 on the moral foundation constructs (Care, Equality, Loyalty, Authority, Proportionality, Purity), the project's core constructs. This is the empirical backing for the Atari-directed thesis that LLMs cannot capture some psychological variables, positioned as a nuance extension of Rathje et al. 2024.

### Deliverables

Two faculty-facing Word documents produced from the characterization run: a four-page deep dive and a one-page executive summary.

---

## 2026-05-28, OrangeGrid Job 469622 Outcome Review: Locked Throughput, Eval Baseline, and V1 Experiment Prep

Session focused on reviewing the outcome of job 469622 (submitted 5/21, finished 5/21 14:46:53 EDT) ahead of the weekly Introne meeting. Output file location, locked throughput, eval against the gold standard, and current state of V1 experiment preparation. Also reconciled the 5/21 meeting transcript with the running project state, particularly the throughput framing.

### Job 469622 Outcome

The 100-text test run completed cleanly. ExitCode 0, total wall-clock 7125 seconds (118.7 minutes). vLLM startup took 200 seconds (faster than the 600-second timeout budget and faster than the 3.5-minute first-request guess from the live snapshot). All 7,500 expected API calls served (100 texts x 3 prompts x 25 labels). Feather output written with 300 rows and 58 columns.

Steady-state throughput locked at **63 RPM** from the wrapper banner. The 65 RPM figure carried in the 5/21 WORKLOG was a live snapshot at 13:54 with 4,291 of 7,500 calls complete; the final whole-run number printed at job exit is 63. Going forward, 63 is the OG V0-engine baseline number to reference.

Condor history had aged out by 5/28 (the schedd's history file rotated within a week), so JobStatus and timing from `condor_history 469622` returned empty. All lifecycle data was recovered from `logs/test_og.out` and `logs/test_og.log` instead. Worth noting for future post-mortems: do not rely on `condor_history` past a few days, snapshot the relevant ClassAds immediately after job completion if they will be needed later.

### File Location Diagnosis

Initial inspection assumed the Feather would land under `outputs/run_015_config3_test/` per config3.yaml's `output_dir` and `output_prefix` settings. It did not. The actual file landed at `results/test_results.feather` because the HPC pipeline (`camel_annotate_hpc.py`) takes the output path as a positional CLI argument from the wrapper, which gets it from the submit file's `arguments` line, and ignores the config's `output_dir`/`output_prefix` entirely.

This is by design, not a bug. The HPC pipeline is built around per-chunk output paths so that 36 production jobs (12 chunks x 3 models) do not collide on a single shared output filename. The non-HPC pipeline (`run_annotation.py`) reads `output_dir`/`output_prefix` from config and builds a timestamped filename, which is the right behavior for interactive Apophis runs but the wrong behavior for chunked HPC production. Two scripts, two output philosophies, both correct for their use case.

What this does mean is that before the production wave, the per-chunk submit file generator (presumably `split_data.py` or similar) needs verification to confirm it constructs unique output paths per chunk and per model. That is a separate task and not blocking on today.

### Test Results Evaluation

Pulled the Feather from OG to Apophis via rsync push, landing at `~/code_space3/hpc/results/test_results_v0_fp8_run469622.feather` (the rename preserves the V0 baseline identity for future comparison). Ran `evaluate.py` against `samples/CAMEL_cleaned_data_complete_ans.feather`. The evaluator matched 100 of 56,796 gold rows to predictions, with 22 of 25 labels showing at least one gold positive in the sampled 100 texts.

Macro F1 across the three prompting strategies:

| Prompt | Macro F1 | Macro P | Macro R | Micro F1 |
|---|---|---|---|---|
| 0-shot_binary | 0.191 | 0.201 | 0.261 | 0.294 |
| multi-shot_binary_reasoning | 0.196 | 0.178 | 0.299 | 0.301 |
| 0-shot_binary_incentive_strict | **0.208** | 0.204 | 0.288 | **0.302** |

The shape of the result is consistent with the March Apophis benchmark. Incentive-strict is the best prompt, multi-shot offers no meaningful improvement over zero-shot, and the cluster of three prompts is tight (0.191 to 0.208 macro F1). The absolute number is lower than the March 0.285 macro F1 figure (Llama 3.1, 50-sample, incentive_strict on Apophis). Roughly 27 percent relative drop. Several plausible explanations not yet investigated: Llama 3.3 vs 3.1 represents a different post-training distribution and is not directly comparable; the 50-sample vs 100-sample draw introduces variance; config3.yaml uses longer construct definitions than whatever config the March run used, and longer prompts may degrade performance; the three labels without gold positives in this 100-text sample drop out of the macro F1 denominator and that recomputation can shift the headline number. Worth investigating before treating the F1 as a regression, but not before V1 throughput optimization is settled.

### Throughput Reframe per 5/21 Meeting

Reconciling the locked 63 RPM with the 5/21 Introne meeting transcript shifts how the OG slowdown should be characterized in subsequent documents and conversations. Introne explicitly said that he typically runs at about three quarters the speed of Apophis on OG, and listed the contributing factors as slower disk IO on the OG home filesystem versus Apophis's NVMe, HTCondor packing multiple jobs onto the same physical GPU, and per-CPU differences. He treats the slowdown as an expected HPC overhead, not a pathology.

If the Apophis baseline of 195 RPM holds at the API-call level (which is still pending verification via the M8.T10d Apophis re-baseline task), Introne's 75 percent rule projects an OG ceiling of approximately 146 RPM. The current 63 RPM is at roughly 43 percent of that ceiling, meaning there is real optimization headroom but the absolute target is not parity with Apophis. The 3x slowdown framing in WORKLOG_2026-05-21 and in Production_Runtime_Estimates_Rev2.docx is sharper than Introne's actual read and should be softened in any Rev3 estimates document. The V1 engine experiment becomes the natural next intervention because removing the fp8 KV cache flag unblocks the V1 engine and that is the single most likely source of recoverable throughput.

Other 5/21 meeting items worth carrying forward into project state: vLLM is stateless per API call by default and aggressively prefix-caches across calls, so no per-call vLLM reset is needed (this rules out a category of reset-related optimization paths); logging discipline at production scale must include vLLM server log plus stderr plus stdout per job, because some compute nodes are bad and need to be excluded via Requirements ClassAd edits; the default mitigation for held jobs is hold-edit-release rather than rm-resubmit, per Introne's unverified intuition that rm-resubmit increments usage count and deprioritizes the user; and jobs read from the home directory at runtime, so in-flight edits can be picked up via release without a fresh submit.

Introne is also winding down his own OG usage, mostly runs smaller models that do not need A100s, and asked only for pre-launch pings and the ability to request pauses on greater-than-30-GB-VRAM jobs. Coordination overhead is lower than the 5/14 framing implied.

### V1 Experiment Preparation

Made the file edits required to run the V1 experiment but did not submit. Current state of the OG files:

- `camel_annotate_hpc.py` line 209: `--kv-cache-dtype fp8_e4m3` flag is commented out with a marker comment noting the V1 experiment context
- `test_job_og.sub` arguments line: writes V1 output to `results/test_results_v1_nofp8.feather` so V1 and V0 runs do not collide
- V0 artifacts preserved on OG under `_v0_fp8_run469622` suffix (Feather, vLLM server log, wrapper stdout, condor log)

When the V1 experiment is ready to run, a single `condor_submit test_job_og.sub` triggers it. Expected wall-clock if V1 engine is correctly activated: 60 to 80 minutes. The diagnostic signal to watch is whether the vLLM startup banner reports V1 engine selection or falls back to V0. Steady-state throughput target zone is 120 to 150 RPM, which would place OG inside Introne's 75 percent baseline band and effectively close the optimization phase.

### Outstanding Code Hygiene

The OG GitHub PAT is scoped to Contents: Read only, meaning all edits made to `wrapper.sh`, `config3.yaml`, `camel_annotate_hpc.py`, and `test_job_og.sub` during the 5/21 marathon and today's V1 prep live only on OG and have not been pushed back to GitHub. Source of truth has drifted between OG and Apophis. Two paths to resolve: regenerate the PAT with Contents: Read+Write so future commits from OG push normally, or treat OG as a deployment target only and rsync changes back to Apophis where Claude Code commits them upstream. The PAT R+W path is cleaner long-term and should be addressed before production iteration picks up.

### Next Session Priorities

In rough order:

- V1 engine experiment (submission ready; ~70 min wall-clock if V1 activates cleanly)
- Apophis re-baseline (verify 195 RPM is calls per minute, not rows per minute; affects whether 63 RPM is at 43 percent of ceiling or 1.3 percent of ceiling)
- Qwen 2.5 72B and DeepSeek-R1 32B 100-text validation on OG under whichever config wins
- Production_Runtime_Estimates_Rev3.docx with locked numbers and the 75 percent baseline framing
- Investigate the F1 drop versus March (config3 prompt length is the leading hypothesis)
- Regenerate PAT with Read+Write scope, sync OG edits back to Apophis main
- M9 RQ3 scoping and prompt axis citations remain backlogged from the 5/19 priority stack

---

## 2026-05-21, OrangeGrid 100-Text Test: Submission, Multi-Round Debugging, and First Live Inference Data

Marathon session that took the OrangeGrid pipeline from "Day 0 complete" yesterday to a working 100-text Llama test running on an A100 80GB by midday. Twelve distinct gotchas hit and resolved across two clusters' worth of submit-file, container, environment, and inference-stack misconfigurations. Net outcome: the pipeline works end-to-end, but the observed throughput is roughly 3× slower than the Apophis baseline, which forces a substantial revision of the production wall-clock estimates and an open question about V0 vs V1 engine optimization.

### HF Model Cache Download

Started three parallel tmux sessions on OrangeGrid login node for the three validated models. Used `hf download` (the new CLI; `huggingface-cli` is deprecated as of huggingface_hub 1.15). All three downloads completed cleanly:
- casperhansen/llama-3.3-70b-instruct-awq (37GB)
- Qwen/Qwen2.5-72B-Instruct-AWQ (38GB)
- Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ (17GB, note: Valdemardi uploader, NOT casperhansen as the morning Quickstart had it)

Total cache at 95GB, comfortable within the 4TB quota.

### GitHub Clone Path Difficulties

The repo at https://github.com/ShahaanK/code_space3 is private and GitHub deprecated password auth for HTTPS in 2021. First clone attempt failed at the password prompt. Considered three alternatives: rsync from Apophis (worked but pulled in a 200MB+ .venv-camel directory we did not want on OG), Personal Access Token (clean, what we ended up using), or making the repo public (rejected for privacy). Generated a fine-grained PAT scoped to the one repo with Contents: Read permission, cached via `git config credential.helper "cache --timeout=86400"`. Clone landed clean.

### Broken Symlinks in Cloned Repo

The Apophis repo has `chunks/`, `results/`, and `logs/` as symlinks to `/DATA/szkhan/camel/hpc_data/...`. Git preserves symlinks, so these arrived on OG pointing to a `/DATA` path that does not exist on this cluster. Caused the first chunks rsync attempt to fail with "mkdir failed: File exists" (the symlink path exists, but it's not a directory, so rsync cannot create files inside it). Fix: `rm chunks results logs && mkdir chunks results logs` to replace symlinks with real directories. Then rsynced chunks from Apophis (test_chunk.feather + chunk_manifest.json, 65KB total, fast).

### Submit File: Twelve Gotchas Hit and Resolved

Wrote test_job_og.sub adapted from Prof. Introne's test_job.sub template, then iterated through this sequence of failures:

1. **WhenToTransferOutput + ShouldTransferFiles contradiction**: dry-run rejected `should_transfer_files = NO` combined with `when_to_transfer_output = ON_EXIT`. Removed the second line.
2. **Argus opt-in missing**: reverse-analyze on a free A100 slot revealed START expression `(request_gpus >= 1) && wantsArgusNode`. Added `+wantsArgusNode = True`.
3. **request_gpus underscore vs camelCase**: even with Argus opt-in, slots still rejected. Prof's sample submit file has both `request_gpus = N` AND `+request_gpus = N` because the slot's START checks the underscored form while the standard keyword creates RequestGPUs. Added the custom attribute.
4. **wrapper.sh not executable**: job ran, errno 13 "Permission denied" trying to execute the wrapper. Git clone did not preserve the executable bit. `chmod +x wrapper.sh`.
5. **HOME env var unset**: wrapper.sh uses `${HOME}/miniconda3` and has `set -u` so unset variable kills it. HTCondor doesn't inherit submitter env by default. Added `HOME=/home/szkhan` to environment line.
6. **config2 vs config3 confusion**: wrapper.sh had `CONFIG_FILE="config2.yaml"` hardcoded. Production source of truth is config3.yaml per the morning audit. Edited wrapper.sh to `CONFIG_FILE="${CAMEL_CONFIG:-config2.yaml}"` then passed `CAMEL_CONFIG=config3.yaml` via the submit's environment line. Backwards-compatible for Apophis.
7. **YAML apostrophe escaping bug in config3.yaml**: yaml.safe_load crashed at line 490-492. Lines 492 and 494 had unescaped `object's behavior` inside a single-quoted multi-line string. Line 491 had it correctly escaped as `object''s actions`. The May 14 commit (per git history) caught one but missed two. Fixed with `sed -i "s|object's behavior|object''s behavior|g"`.
8. **CUDA_VISIBLE_DEVICES UUID format**: vLLM crashed with `ValueError: invalid literal for int() with base 10: 'GPU-68babd2c'`. HTCondor sets CUDA_VISIBLE_DEVICES to GPU UUIDs for cgroup isolation; vLLM's `device_id_to_physical_device_id` tries to `int()` each entry. Added 6-line block to wrapper.sh that detects UUID format and converts to integer indices.
9. **GPU memory contention on matched node**: vLLM hit `torch.OutOfMemoryError: GPU 0 has total capacity of 79.25 GiB of which 1.06 MiB is free. Process 1019317 has 74.16 GiB memory in use`. HTCondor's partitionable slot bookkeeping diverged from nvidia-smi reality: thought a GPU was free, but a previously-leaked or shared process was occupying 74 GiB of it. Solution: add `(TARGET.GPUs >= 2)` to Requirements to only match nodes where both A100s are free on the parent slot. Strong proxy for "no other job is touching the physical GPUs."
10. **vLLM did not start within 600s timer**: was the symptom of #8 above, since vLLM crashed early but the orchestrator's health-check loop kept polling for 10 minutes before giving up.
11. **on_exit_hold left held jobs blocking ssh access**: `condor_ssh_to_job` requires a running job. Held jobs cannot be inspected via ssh. Diagnostic flow had to lean on log files (vllm_server.log specifically) instead.
12. **GitHub case-sensitivity false alarm**: Prof's repo URL is github.com/ShahaanK with capital K, but git auth is case-insensitive on the username. Not actually an issue, just a moment of confusion.

### Live Run Data: Job 469622

Submitted at 12:44 EDT. Matched to OG-NODE-10-5-205-213 (one of the both-GPUs-free A100 nodes from the condor_status snapshot). Wrapper banner, conda activation, CUDA_VISIBLE_DEVICES conversion (UUID → 0), vLLM startup all clean. First annotation request at 12:48:05, meaning vLLM startup took 3.5 minutes total (substantially faster than my 10-minute conservative budget).

At 13:54 EDT, 65.8 minutes into inference, the run state is:
- 4,291 of 7,500 API calls completed (57.2%)
- Effective request rate: 65.2 req/min (1.09 req/s)
- Avg prompt token rate: 753.6 tok/s
- Avg generation token rate: 41.3 tok/s (bursty, time-weighted; peak ~108 tok/s when not prompt-bound)
- GPU KV cache usage: 1.6% to 5.7% (well within headroom)
- Concurrent requests: 5 to 16 (target 16, drops during prompt-heavy bursts)

Estimated completion around 14:43 EDT, total wall-clock ~118 min from submit.

### The 3× Slowdown Versus Apophis

Apophis March benchmark for Llama 3.3 70B was 195 req/min on dual A6000 TP=2 with vLLM 0.16.0. Observed OG rate is 65 req/min on single A100 80GB TP=1 with vLLM 0.10.1.dev V0 engine. The A100 is per-chip faster than the A6000 and should compensate for losing TP=2 parallelism, so the 3× slowdown is unexpected and not yet explained.

The most likely contributor is the V0 vs V1 engine difference. vLLM 0.10.1.dev auto-fell-back to V0 because `--kv-cache-dtype fp8_e4m3` is not supported by V1 in this version. V1 is the modern, more-optimized engine; V0 is the legacy path. Removing fp8 KV cache from the vLLM invocation would unblock V1. The tradeoff is roughly 2× KV cache memory per request, but with 80GB headroom this is comfortable.

Other plausible contributors: config3.yaml has more verbose construct definitions than config2.yaml (more prompt tokens per request → more prompt-processing time per call); prefix caching may be less effective in V0 on 0.10.1.dev than it was in V1 on 0.16; 16-thread concurrency may be undersubscribing the A100 (Apophis benchmark concurrency unknown).

### Critical Arithmetic Correction to Earlier Estimates

The morning runtime doc (Production_Runtime_Estimates.docx, Rev 1) and all earlier estimates treated each (text × prompt) pair as one "request." Investigation showed the orchestrator actually makes 25 separate API calls per (text × prompt), one per psychological construct label, parallelized across 16 threads. Total API calls per 100-text test: 100 × 3 × 25 = 7,500 (not 300 as the earlier estimates assumed).

This 25× multiplier was missing from the morning estimates. Even at Apophis baseline rates, the morning projections were 25× too optimistic. Production wall-clock estimates have been revised in Production_Runtime_Estimates_Rev2.docx accordingly.

### Production Wall-Clock at Current Observed Rate

For the full 56K-text corpus across all 3 models on OrangeGrid at the current 65 RPM observed rate:
- Per chunk (5,000 texts, 375,000 API calls): ~96 hours (4 days)
- Single model, 12 chunks parallel on 12 A100s: ~4 days wall-clock
- Three models in parallel, sharing the 18-GPU A100 pool with Introne: 8 to 14 days wall-clock

This is dramatically longer than the morning estimate of 3.7-4.4 hours total. The morning estimate was wrong due to the 25× arithmetic error AND the unobserved 3× slowdown. The Rev 2 doc walks through this in detail.

### Open Questions for Tomorrow

1. **V1 engine optimization viability**: 30-60 minute experiment. Remove `--kv-cache-dtype fp8_e4m3` from the vLLM invocation in `camel_annotate_hpc.py`, re-run the 100-text test, measure RPM delta. If V1 doubles throughput to ~130 RPM, full-corpus wall-clock halves to 4-7 days. High-value test.
2. **Apophis re-baseline**: confirm the 195 RPM was measured at the API-call level (not the row level). If Apophis was actually 195 rows/min, the calls-per-minute equivalent is 4,875, making OG far further behind than 3× and changing the optimization priority.
3. **config3 vs config2 prompt token impact**: re-run 100-text test on config2 to isolate whether the longer construct definitions in config3 contribute to the slowdown.
4. **Concurrent thread count**: try 32 or 64 threads in the wrapper, see if RPM scales.

### Documents Produced This Session

- `test_job_og.sub`: OrangeGrid HTCondor submit file with all 12 gotcha fixes baked in
- Modified `wrapper.sh`: CAMEL_CONFIG env var override added, CUDA_VISIBLE_DEVICES UUID→integer conversion block added
- Modified `config3.yaml`: two unescaped apostrophes fixed (lines 492 and 494)
- `Production_Runtime_Estimates_Rev2.docx`: revised production estimates using live OG observed rate, with explicit acknowledgment of the morning estimate's arithmetic error and the V0 engine slowdown

### Next Session Priority Stack

(a) Wait for job 469622 to complete (expected 14:43 EDT today), lock in final RPM number
(b) V1 engine optimization experiment (open question 1 above)
(c) Apophis re-baseline (open question 2 above)
(d) If V1 helps: re-run 100-text test on Qwen and DeepSeek with V1 engine
(e) If V1 helps significantly: proceed to 1,000-text characterization wave on all 3 models
(f) Send Introne the revised wall-clock estimates + optimization roadmap before scheduling production
(g) Update Production_Runtime_Estimates_Rev2.docx to Rev 3 with post-optimization numbers


---

## 2026-05-19: Day 0 OrangeGrid Environment Setup (M8 Plumbing)

Day 0 of the M8 production-setup sequence on OrangeGrid. Goal was to land the vLLM Singularity container Prof. Introne provided, stand up a Python environment per the SU OrangeGridExamples canonical pattern, and verify vLLM is importable before any GPU work begins. Day 1 (HF model weight pre-cache) and Day 2 (batch script adaptation, 100-text test) intentionally deferred so each checkpoint clears cleanly before the next.

### SIF Delivery and Transfer

Prof. Introne SCP'd `vllm-openai.sif` (11GB, owner `jeintron`, build timestamp May 14 08:55) to `/DATA/vllm-openai.sif` on Apophis during the 5/14 meeting. He couldn't write into `/DATA/szkhan/` so it sits at the `/DATA` root; left untouched there as a backup. Push-SCP from Apophis directly to OrangeGrid `/home/szkhan/` succeeded on the first try. On-cluster SIF mtime is May 19 18:52, matching the transfer window. The 4TB NFS quota easily absorbs the 11GB plus the ~92GB of model weights coming on Day 1.

`singularity inspect` on OrangeGrid (Singularity 3.7.1) confirmed three things worth recording: the image is `from: vllm/vllm-openai:v0.10.0` (no custom layers, upstream Docker base), built with Singularity 3.7.1 (matches OrangeGrid version, no compatibility skew), and dated Tuesday December 9 2025. The expected sha256sum verification step was skipped because `singularity inspect` reading the SIF header cleanly is itself sufficient corruption detection.

### Miniforge per SU Canonical Pattern

Followed the SU OrangeGridExamples `/python/README.md` install path verbatim. Miniforge 24.7.1 (pypy3 build) downloaded from the conda-forge GitHub release, installed batch-mode to `$HOME/miniconda3`. Conda init wrote the activation block to `.bashrc`. The `.bash_profile` shim that sources `.bashrc` was added per the SU instructions, since OrangeGrid login shells read `.bash_profile` rather than `.bashrc` by default. After a logout/login cycle, `which conda` returned `/home/szkhan/miniconda3/bin/conda` and `conda 24.7.1` as expected.

Deviation from my earlier guidance worth noting: my initial plan called for vanilla Miniconda from Anaconda's repo. The SU example file `python/README.md` explicitly uses Miniforge (conda-forge default channel) instead. Switched to that on-the-fly. Functionally equivalent for our purposes, but if any package solver issues arise later they will be conda-forge sourced, not the Anaconda default channel.

### camel Environment

Created `camel` conda env with Python 3.11 to match Apophis `~/myenv`. Pip-installed pandas, pyarrow, pyyaml, huggingface_hub, openai, requests. Notable versions: huggingface_hub 1.15.0, openai 2.37.0, pandas 3.0.3, pyarrow 24.0.0, pydantic 2.13.4. No vLLM or torch in the conda env on purpose; those live inside the SIF.

### vLLM Version Verification (with Two Gotchas)

First attempt at `singularity exec --nv ... python -c "import vllm; ..."` failed two ways. The `INFO: Could not find any nv files on this host!` warning is expected on the login node which has no GPU drivers, harmless. The `FATAL: "python": executable file not found in $PATH` is a real trap: the `vllm/vllm-openai` Docker image ships `python3` but not a `python` symlink. Re-ran with `python3` and dropped `--nv` (not needed for an import-only check):

```
singularity exec /home/szkhan/vllm-openai.sif python3 -c "import vllm; print(vllm.__version__)"
0.10.1.dev1+gbcc0a3cbe
```

Important nuance for the methods section. The SIF is **not** pristine vLLM 0.10.0. It is `0.10.1.dev1+gbcc0a3cbe`, a development snapshot Prof. built off the 0.10 branch at commit `bcc0a3c`. Functionally identical to 0.10.0 release for our purposes (same OpenAI server, same prefix caching mechanism, same AWQ kernel paths) but should be cited as `vLLM 0.10.1.dev (commit bcc0a3c)` in the paper, not 0.10.0. Apophis remains pinned at vLLM 0.16.0; the version delta across clusters is now a documented methodological choice, not an oversight.

### Cluster Topology Delta to Record

The two-cluster split now has a real version axis. Apophis runs vLLM 0.16.0 on CUDA 12.x for dev/test. OrangeGrid runs vLLM 0.10.1.dev on CUDA 12.8 for production. The lower vLLM version is forced by CUDA 12.8 being the highest driver available on OrangeGrid (recent vLLM snapshots target CUDA 12.9). The prefix-caching system/user prompt split delivered +7-9% RPM on Apophis at vLLM 0.16; whether 0.10.1.dev preserves that gain is an empirical question to settle with the 100-text test on Day 2. Path forward already chosen: run `--no-split` legacy mode first to baseline RPM, then re-run with split enabled, log the delta.

### Open Items Heading Into Day 1

HF model weight pre-cache on the OrangeGrid login node (Llama 3.3 70B AWQ, Qwen 2.5 72B AWQ, DeepSeek-R1 32B AWQ; ~92GB total). Should run inside tmux given RDS session fragility. Repo clone and chunks rsync from Apophis. SIF deep-dive via Claude Code to recover entrypoint, env vars, and AWQ kernel info needed for the batch wrapper. Batch script adaptation for OrangeGrid: TP=1, A100 80GB constraint, HF cache bind, `python3` not `python` in any container invocation. 100-text Llama test deferred to Day 2 with target of running before the 5/21 weekly meeting if Day 1 closes cleanly.

---

## 2026-05-14: Weekly Meeting with Prof. Introne (Thursday 8:30 AM)

Short, dense meeting focused entirely on unblocking the M8 OrangeGrid setup. Five concrete pieces of guidance came out of it, plus a coordination protocol Prof. Introne explicitly asked for, plus a side conversation about Prof. Banks's local-deployment use case that connects back to the DGX Spark deliverable. The CAMEL-relevant content is logged here; the Banks-side consulting thread is separate.

### vLLM Version Constraint on OrangeGrid

The first agenda item was confirming vLLM works on OrangeGrid at all (yes) and resolving the version delta against Apophis. Prof. Introne stated directly: recent vLLM snapshots target CUDA 12.9, OrangeGrid only has CUDA 12.8, and vLLM 0.10 is the highest version he found that runs on the earlier CUDA. This locks the two-cluster version split: Apophis stays at 0.16.0 (validated March pipeline, do not touch), OrangeGrid runs whatever 0.10.x build is inside the SIF he provides. Reinstalling vLLM on Apophis to match OrangeGrid was not discussed and would violate the existing pinned-environment hard constraint.

### SIF Delivery Offered and Executed

Prof. Introne offered to bypass the "pull vLLM Singularity container from Docker Hub on the OrangeGrid login node" step entirely by copying his existing `vllm-openai.sif` over. He SCP'd the 11GB image to `/DATA/vllm-openai.sif` on Apophis during the meeting (couldn't write to `/DATA/szkhan/` since he doesn't own that directory). The image is portable: copy to OrangeGrid, no build step needed. This kills M8.T6 (pull SIF on OrangeGrid login node) as a discrete task and removes the bandwidth question for that step.

### A100 Access Confirmed Operational

Reported that ITS had resolved the A100 visibility issue from the 5/7 recon. The pool is now the documented 27 nodes / 68 GPUs including 9 A100 80GB nodes, all visible to the szkhan account. The locked strategy of single-A100 TP=1 for 70B/72B models holds; TP=2 across L40S/A40 pairs is officially shelved as a fallback that no longer needs to be planned for.

### HuggingFace Cache and Container Bind Pattern

Prof. Introne walked through the canonical pattern for HF model weights: download weights into a local cache directory on the OrangeGrid login node using `huggingface_hub`, then have the Singularity container access them via a bind mount. Pasted the relevant code snippet from his fraction code into chat: `HF_CACHE = os.path.expanduser("...")` followed by `singularity exec --bind $HF_CACHE:$HF_CACHE`. This is the M8.T9 batch script requirement. Compute nodes do have outbound internet access ("they can pull, but you can't push") so HF downloads from compute would also work, but pre-caching to `/home/szkhan` remains the right call for speed and reliability.

### GPU Memory Advertising and Coordination Protocol

Prof. Introne emphasized that HTCondor allocates jobs to GPU slots based on the memory each job advertises. Under-advertising and over-consuming causes the over-consuming job to clobber concurrent jobs on the same physical GPU. The locked single-A100 strategy is comfortable on this front: 37GB AWQ weights plus KV cache headroom fits inside 80GB with margin.

The coordination protocol is the new commitment from this meeting. Prof. Introne is about to launch his own work on the cluster: first a large CPU-only clustering job set, then a GPU job. He explicitly asked that production CAMEL runs go out in subsets rather than all at once so per-chunk runtime can be characterized, and that timing estimates be reported back to him so he can sequence his own grant work around the CAMEL load. Stated three-day window for getting the test + production launched. Logging this as M8.T15 (subset reporting) and M8.T16 (ongoing coordination pings).

### RQ2 Prompt Axes Deferred to Second Wave

Sequencing decision Prof. Introne agreed with: the first 56K production wave runs with the four prompting strategies already validated in the March pipeline (zero-shot, few-shot, minimal, incentive_strict). The new culturally-grounded prompt axes from the 4/23 RQ revision (warmth, power/authority, guilt, transactional) are an add-on after the base wave completes. Rationale is to lock the base baseline first and extend from there. This is not a milestone change so much as a sequencing note, but worth flagging in M7 because it affects when RQ2 has data to support it.

### Side Items Disposed

DGX Spark research doc was discussed briefly. Prof. Introne hasn't seen the draft yet. Committed to resending after the meeting; that has now been done. The Banks-side consulting conversation (Jamie's local-LLM setup for chatbot replication research) is separate from CAMEL and not tracked here.

### Outcome

M8.T6 is functionally complete pending the SIF copy. M8 sequence can proceed without further blocking dependencies from Introne. The 5/21 weekly meeting should be a status update with concrete timing numbers from the 100-text test, not another planning session.

---

## 2026-05-11 — ITS A100 Visibility Expanded, Strategy Locked to Option A

### ITS Research Computing Response

The 5/7 email to researchcomputing@syr.edu (cc Prof. Introne) asking whether A100s exist in a partition not visible to the szkhan account got resolved. ITS expanded our pool visibility, and the 5/11 re-run of the GPU inventory query returned nine new A100 80GB PCIe nodes (2 GPUs each, 18 A100s total, driver 12.8, CUDA capability 8.0, 81154 MB usable VRAM per card). Prof. Introne was correct that A100s existed on OrangeGrid; the 5/7 invisibility was an account permissions configuration that ITS resolved rather than a missing hardware claim.

The expanded visibility also surfaced four additional A40 nodes (now 7 total instead of 3 from the 5/7 recon, all under the OG-NODE prefix), one Quadro RTX 5000 node, and two GeForce GTX 1080 Ti nodes. Three CRUSH-NODE A40 machines from the 5/7 recon (CRUSH-NODE-10-5-15-2, 56-2, 82-4) disappeared from the queryable pool. Cause unknown; they may be drained for maintenance, moved to a different group, or fallen out of view as a side effect of the permissions change. Worth a one-line follow-up to Research Computing eventually but not urgent given the much larger pool now visible.

### Updated GPU Pool Inventory

Total visible pool is now 60 nodes / 172 GPUs across six device families. Usable for vLLM 0.16 (requiring CUDA 12+ driver): NVIDIA A100 80GB PCIe (9 nodes, 18 GPUs, 80 GB VRAM, driver 12.8), NVIDIA A40 (7 nodes, 28 GPUs, 45 GB VRAM, driver 12.0), NVIDIA L40S (11 nodes, 22 GPUs, 45 GB VRAM, driver 12.8). Effective usable pool: 27 nodes, 68 GPUs. Unusable: 29 Quadro RTX 6000 nodes (driver 11.6), 1 Quadro RTX 5000 node (Turing, similar driver age), 2 GeForce GTX 1080 Ti nodes (Pascal, far too old).

The driver-version check on the A100 ClassAd returned `CUDADriverVersion = 12.8` and `CUDAMaxSupportedVersion = 12080`, which is the same as the L40S nodes. No vLLM compatibility blocker on A100.

### Strategy Locked: Option A — Single-GPU A100 per 70B Model

With A100 80GB confirmed available and confirmed CUDA-12.8 compatible, Prof. Introne's 4/23 batch-script advice becomes valid again. The TP=2 strategy proposed on 5/7 (across L40S/A40 pairs) is shelved as a fallback contingency to be activated only if A100 contention becomes a production blocker. The committed strategy is:

- 70B and 72B models (Llama 3.3, Qwen 2.5, AceGPT-v2): TP=1, single A100 80GB, requirements expression targets `CUDADeviceName == "NVIDIA A100 80GB PCIe"`. 37 GB of weights leaves ~43 GB headroom for KV cache and prefix caching, comfortable for production-scale annotation.
- 32B and 24B models (DeepSeek-R1, Mistral Small, Falcon-H1): TP=1, single GPU, either A100 or L40S/A40 depending on availability. 17-18 GB weights at INT4 fits comfortably on any of these.

Capacity outlook: if the cluster were entirely free, 18 simultaneous 70B jobs could run on A100 alone. Real-world contention will reduce this, but the ceiling is high enough that 12 chunks per model × 3 models = 36 jobs should land in roughly two or three waves, not many more.

### 5/7 Hello-World Test Resolved

The 5/7 hello-world submission resolved as follows. Clusters 396606, 396607, 396608 were removed during the iterative constraint refinement that day. Cluster 396609.0 completed successfully on 5/7 at 09:05 with a 1-second runtime — confirming the submission pipeline works end to end. The output file was not found in the home directory on 5/11 lookup, but the `condor_history` evidence of clean completion is sufficient and the run itself is not blocking anything.

### Documentation Updates

Memory entries 10, 16, and 17 replaced to reflect the new pool reality, the locked strategy, and the resolved A100 access. WORKPLAN changelog gained a dated 5/11 block. WORKPLAN cluster topology table body still needs a cleanup pass to remove the 5/7 stale entries; deferring to the next session.

### Open Items for Next Session

The next milestone is M8 environment setup with the locked strategy. The sequence is to pull `vllm-openai.sif` Singularity container to `/home/szkhan`, pre-cache the three validated model weights (Llama 3.3 70B AWQ, Qwen 2.5 72B AWQ, DeepSeek-R1 32B AWQ) to `/home/szkhan`, clone the camel code repo to OrangeGrid, adapt the OrangeGrid batch script for single-GPU A100 (TP=1, `request_gpus=1`, A100 device constraint, no tensor parallelism), submit a 100-text test job for Llama 3.3 70B on a single A100, and only after that test clears proceed to 56K production submission. Apophis 100-text test (M4.T11) also still pending and not advanced today.

---

## 2026-05-07 — OrangeGrid First Recon, GPU Pool Characterization, TP=2 Strategy Revision, Weekly Meeting

### OrangeGrid First SSH and Empirical Recon

First successful SSH into OrangeGrid via SU RDS to `szkhan@its-og-login5.syr.edu`. Login banner reports Ubuntu 20.04.4 LTS, Condor 9.8.0 (April 2022 build, older than current 10.x releases but stable). Home directory at `/home/szkhan` is NFS-mounted from NetApp `10.5.0.207`, quota 3.6 TB soft / 4 TB hard (well over the ~120 GB needed for three model AWQ caches). No write activity yet — recon was stdout-only, no filesystem writes to the user home.

Ran an empirical recon block covering login sanity, Condor configuration defaults, runtime and preemption policy, GPU pool inventory, ClassAd attribute discovery, Singularity availability, module system, scratch storage, and submission capability. Key infrastructure findings: `PREEMPT = False` means jobs are not evicted by competing higher-priority work, which removes the main worry about long-running Jupyter sessions on compute nodes. `JOB_DEFAULT_REQUESTDISK = DiskUsage` confirms Prof. Introne's "drop the explicit request_disk" guidance from 4/23 was correct. Singularity 3.7.1 and Apptainer 1.1.3 are both installed. No Lmod or Environment Modules (`module: command not found`), meaning batch scripts must be self-contained. Compute-node scratch is `/var/lib/condor/execute/` and ephemeral per job, so the M8.T7 plan to pre-cache weights to `/home/szkhan` is the correct path.

### GPU Pool Characterization — No A100s Visible

Three GPU families visible in the OrangeGrid pool, no A100s anywhere. Three NVIDIA A40 nodes (4 GPUs each, 45 GB usable VRAM, CUDA driver 12.0). Eleven NVIDIA L40S nodes (2 GPUs each, 45 GB usable, CUDA driver 12.8). Twenty-nine Quadro RTX 6000 nodes (mixed 2-4 GPUs each, 22 GB, CUDA driver 11.6). Total visible fleet: 43 machines, 130 GPUs.

The Quadro RTX 6000 fleet is almost certainly unusable for our pipeline because driver 11.6 corresponds to CUDA toolkit 11.x, and vLLM 0.16 requires PyTorch built against CUDA 12. Effective pool for our work is therefore A40 + L40S only: 14 machines, 34 GPUs.

A100 search was performed three ways: VRAM filter at >= 70000 MB returned empty; regexp on `CUDADeviceName` matching "A100" returned empty; full distinct device list across all advertising slots showed only A40, L40S, and Quadro RTX 6000. Either A100s are not on this cluster, or they sit in a partition our account cannot see. The 4/30 WORKLOG entry's mention of A100 80GB was based on Dan/ITS's verbal description of the cluster topology and turned out to be either inherited assumption or stale information.

Recon also surfaced a Condor 9.8 quirk: the GPU ClassAd attributes use a `CUDA` prefix (`CUDADeviceName`, `CUDAGlobalMemoryMb`, `CUDAClockMhz`, `CUDADriverVersion`) rather than the `GPUs_` prefix used in Condor 10.x. Initial queries failed because they used the newer schema. Worth documenting in the WORKPLAN reference table.

### Strategy Revision: TP=2 Across L40S/A40 Pairs

Prof. Introne's 4/23 batch-script advice (target A100 80GB, single GPU per model, no tensor parallelism) was built on the premise that A100s would be available. Without them, single-GPU 70B AWQ on a 45 GB L40S or A40 leaves only ~8 GB of headroom after the weights load, which is too tight for production-scale KV cache and prefix caching.

The cleaner play is TP=2 across two L40S or two A40 GPUs on the same node, giving 90 GB of total VRAM. This is essentially the same envelope as the Apophis-validated TP=2 config (96 GB across two A6000s), so the configuration risk is low. The proposed split: 70B/72B models (Llama 3.3, Qwen 2.5, AceGPT-v2) run TP=2 with `request_gpus = 2`; 32B/24B models (DeepSeek-R1, Mistral Small, Falcon-H1) run TP=1 with `request_gpus = 1`.

If the cluster were entirely free, the L40S and A40 fleets could support roughly 17 simultaneous TP=2 jobs (3 A40 nodes × 2 TP=2 jobs per node + 11 L40S nodes × 1 TP=2 job per node). It is not free. Every visible usable GPU was claimed by other users at the time of recon. For the 56K production runs, jobs will queue and run in waves as slots free up overnight. Rough back-of-envelope wall time is around 10 hours per model under decent availability.

### Submission Pipeline Validated, Hello-World Queued

A test job was prepared (`~/test_nvidia_smi.sub`) to validate the submission pipeline end-to-end. First submission attempt failed because `nvidia-smi` is not installed on the login node and Condor checks executable existence on the submit side before queuing. Fix was adding `transfer_executable = false` so Condor trusts the executable will exist on the target compute node.

Second submission cleared (cluster 396606), sat idle, and `condor_q -better-analyze` correctly diagnosed that no L40S slots could match because all partitionable parent slots had GPUs = 0 free (every GPU carved out to dynamic children, all of which were claimed by other users). Same pattern on A40 nodes when we widened the constraint. The submission pipeline itself is fully validated: queue accepts jobs, constraints evaluate correctly, the analyzer explains rejections clearly.

After Prof. Introne's meeting (see below), the constraint was relaxed to allow any of the three GPU types and memory request was lowered to 512 MB. The current job (cluster 396607 or higher) is sitting in queue waiting for any GPU slot anywhere to free up. Outcome will appear in `~/test_nvidia_smi.<cluster>.0.out` when something frees.

### Weekly Meeting with Prof. Introne (8:30 AM)

Presented the TP=2 strategy and the A100 recon findings. Prof. Introne pushed back on the A100 absence, asserting that A100s do exist on OrangeGrid for research use. Did not provide a specific partition name, pool hostname, or access path. Action item from the meeting was to send a query to ITS Research Computing confirming whether A100s exist in a partition or pool not visible to the szkhan account by default.

Other carry-overs from prior meetings (formal RQ structure sign-off, GLOBE preprint citation, prompt axis citations) were not advanced in this meeting because the OrangeGrid strategy question dominated the agenda. TP=2 strategy is conditionally accepted pending the A100 question's resolution.

### ITS Email Sent

Drafted and sent an email to `researchcomputing@syr.edu` (cc'd Prof. Introne) framing the A100 question as a visibility/access query rather than a "you missed something" challenge. References Dan's prior 4/29 access approval for continuity. Asked three specific questions: are there A100s on OrangeGrid invisible to my account by default; if so what is the access path; and is there a separate Condor pool I should be querying with `-pool`. Response pending.

### Documentation Updates

The 4/30 WORKLOG entry was extended with a "Tail-End Discussion" subsection capturing the safety-axis-closed decision, the M9 reading direction expansion (Qwen + DeepSeek + Nemotron post-training docs), and the directional ablation clarification. Corresponding changelog line added to WORKPLAN for the 4/30 block. Both were committed earlier this session before the recon began.

A meeting-prep Word document was generated as a one-page bulleted brief for the 8:30 AM meeting (`Meeting_Prep_2026-05-07.docx`). All-black text, no em-dashes, conversational tone, structured around the TP=2 ask.

### Open Items Going Into the Next Session

The hello-world test is still queueing. The vLLM Singularity container has not been pulled to OrangeGrid. The model weights have not been pre-cached to `/home/szkhan`. The code repo has not been cloned. The A100 question is pending an ITS response. The Apophis 100-text test (M4.T11) was not advanced today.

---

## 2026-04-30 — Weekly Meeting with Prof. Introne, HPC Strategy Clarification, OrangeGrid Approval, Memory & Documentation Realignment

### Meeting Summary (Thursday 8:30am)

Two-cluster picture finally surfaced clearly. Until today, the project mental model conflated "Apophis" and "the HPC cluster" because Prof. Introne had been describing both interchangeably across earlier meetings. Today established that Apophis is Prof. Introne's lab box (no Condor scheduler), and the institutional HPC he has been coaching us toward is the Syracuse ITS-managed OrangeGrid cluster, which Prof. Introne does not control. His 4/23 batch-script guidance — single GPU per model, no tensor parallelism, drop disk requirement, target A100s — was for OrangeGrid all along. Applying it to Apophis on 4/30 morning broke the validated TP=2 configuration and was reverted same day.

### OrangeGrid Access Approved

ITS Research Computing (Dan) approved OrangeGrid access on 4/29. Login node: `its-og-login5.syr.edu`. Connection via SU RDS. The cluster is a heterogeneous HTC pool (HTCondor scheduler) with hundreds of GPUs across SUrge infrastructure (A100 80GB, L40S 48GB, A6000, A40, mixed RTX). Storage is NetApp-backed `/home/<netid>` (4 TB default), automatically mounted on compute nodes — no file transfer needed for jobs. Login nodes are explicitly submit-only: no Jupyter, no IDE, no long tmux sessions.

This is materially different from Apophis in every way that matters for a job submission strategy: scheduler exists, storage is shared NFS, GPU pool is heterogeneous and opportunistic, and runtime/preemption rules are governed by ITS rather than Prof. Introne. The empirical recon plan (M8.T3) is to log into OrangeGrid and run a sequence of diagnostic Condor commands to discover ClassAd attributes, default disk requests, and runtime limits — answering most ITS questions without an email roundtrip.

### Apophis Configuration Reverted

Earlier in the day, before the cluster picture was clear, batch scripts in `~/code_space3/hpc/generate_batch_jobs.py` were edited to apply Prof. Introne's 4/23 guidance: changed all 70B models to `tp=1`, `request_gpus=1`, `gpu_vram_mb=70000`. This broke the validated configuration: Llama 3.3 70B AWQ at INT4 needs ~37 GB and ran successfully on Apophis with TP=2 across both A6000s in March (run_005, run_007). Single-A6000 (48 GB) is tight on KV cache headroom and was never tested.

After the meeting clarified the cluster split, the registry was reverted: 70Bs back to `tp=2`, `request_gpus=2`, `gpu_vram_mb=40000`. The 32B/24B entries (DeepSeek, Mistral, Falcon-H1) remained at `tp=1`. The OrangeGrid-style config will be re-applied later when the M8 OrangeGrid environment is stood up — but as a separate config path, not a replacement for Apophis.

### Apophis Test Job — Manifest Issue

The 100-text test chunk run on Apophis required hand-writing `chunks/chunk_manifest.json` because `--create-test` mode in `split_data.py` does not generate one. After three iterations to match the `split_data.py` writer schema (added `chunk_size`, then `total_unprocessed`, then `rows`/`original_start_index`/`original_end_index` per chunk), `generate_batch_jobs.py --model llama-3.3-70b` succeeded and produced `batch_llama-3.3-70b.sub` and `wrapper_llama-3.3-70b.sh`. Submission deferred until tomorrow because Apophis isn't actually running Condor anyway — direct vLLM via the wrapper is the cleaner path.

### RQ3 Viability Scoping Plan

Per the 4/23 meeting and Prof. Introne's caveat that RQ3 is conditional on finding clean HuggingFace model lineages with documented post-training variants, a 4-hour timebox was set aside for this scoping (M9). Candidate families in priority order: Allen AI OLMo + Tülu (most likely hit, fully documented SFT/DPO/RLHF on the same base), Nvidia Nemotron (Prof's suggestion, fully open training data), Llama 3 base + community fine-tunes, Qwen base/Instruct/Chat documentation, Zephyr/HuggingFaceH4 (well-documented DPO vs SFT). Decision criterion is binary: if a clean lineage is found, RQ3 lives; otherwise fold its spirit into RQ1 and drop.

### Memory Cleanup & Documentation Realignment

Project memory was overgrown with stale entries from earlier weeks and contained two duplicate model roster entries after a session edit. Cleanup pass: dropped #8 (prompt strategy literature — moved to WORKPLAN reference table), dropped #12 (self-referential WORKLOG/WORKPLAN existence note), merged old research-axes entries into a single 4-RQ-grounded entry, and added new entries for: Apophis 70B AWQ validation status, OrangeGrid setup status, two-config HPC distinction. Memory now sits at 18 entries (down from 23), with 12 slots free for the OrangeGrid setup phase.

WORKPLAN was rebuilt from scratch with M8 (OrangeGrid Setup & Production Runs) and M9 (RQ3 Viability Scoping) added as new milestones. Stale states corrected throughout: M2.T14, M3.T12, M3.T13 marked done (completed in milestone doc submitted Mar 12). M4 renamed to "Apophis Dev/Test Pipeline" with a note about the cluster clarification. M4.T10 added to record the Apophis 70B AWQ validation explicitly. M4.T11 added for the current test_chunk validation. M5 explicitly flagged BLOCKED ON BUDGET APPROVAL (~6 weeks). M6.T7–T11 added covering the prompt valence lit deep-dive (Apr 2), training corpora research (Apr 2), GLOBE preprint hunt, Jinfen-aligned architecture lit, and prompt axis citations. M7.T1c added for the RQ restructure (5 → 4); M7.T2 marked NOT NEEDED; M7.T7 deadline marked OPEN-ENDED. Reference tables refreshed: model roster split into validated-vs-planned per cluster; cluster topology table added (Apophis vs OrangeGrid); cost summary table flagged PENDING UPDATE; new prompt axes literature table added.

### DGX Spark Deliverable

Local-deployment deep dive promised to Prof. Introne for the businessman side conversation. Full landscape report drafted covering DGX Spark, Mac Studio M3 Ultra, GPU workstations (RTX 5090, dual 5090, RTX PRO 6000 Blackwell, used 3090 builds, pre-built integrators), and cloud baseline (per-token APIs, GPU rentals, serverless inference). Three-angle analysis per option: performance/capability, total cost of ownership, setup complexity/ecosystem/lock-in. Decision matrix by use case and budget. Honest "when not to go local" section. Pending share to Prof. Introne.

### Tail-End Discussion: Fine-Tuning Literacy and Safety Axis Closed

After the cluster-strategy and OrangeGrid items wrapped, the meeting closed with a short discussion on last-mile fine-tuning (RLHF, constitutional AI) and where safety restrictions live in the model lifecycle. Prof. Introne floated safety protocols as a possible additional axis of variation alongside training corpus, prompt framing, and architecture, then immediately walked it back: "we've already got so much going on, it's probably not worth it." **Decision: safety / fine-tuning policy NOT added as an analytical axis.** Logged so it doesn't resurface.

For M9 (RQ3 viability scoping), Prof. Introne pointed to the more open model providers (Qwen, DeepSeek, Nvidia) as better sources of post-training methodology documentation than OpenAI or Anthropic. Anthropic was acknowledged as decent but less forthcoming than the open-weight teams. Shahaan confirmed Nemotron is on the M9 list (M9.T2) and will incorporate reading post-training method docs into the scoping, not just verifying variant existence on HuggingFace.

Directional ablation clarified as background literacy: ablation = zeroing out neurons or removing training data; directional ablation = fine-tuning with a loss function aimed at moving the refusal vector in the opposite direction. Prof. Introne hedged on the exact mechanism. No research direction implied.

### Action Items

- [ ] Submit Apophis 100-text test job using reverted TP=2 config
- [ ] First SSH into OrangeGrid via SU RDS
- [ ] Run empirical Condor recon on OrangeGrid login node
- [ ] Email researchcomputing@syr.edu for any gaps recon doesn't answer (cc Introne)
- [ ] Stand up OrangeGrid environment (conda/Singularity per ITS conventions)
- [ ] Pre-cache HuggingFace model weights to OrangeGrid `/home/szkhan`
- [ ] Adapt batch scripts for OrangeGrid (separate config path from Apophis)
- [ ] OrangeGrid 100-text test job
- [ ] Full 56K production runs across all 3 validated models in parallel
- [ ] RQ3 viability scoping (4-hour timebox, M9)
- [ ] Send DGX Spark deep-dive to Prof. Introne
- [ ] Get formal RQ structure sign-off from Prof. Introne
- [ ] Locate LLM-as-jury GLOBE framework preprint (Apr 23 reference)
- [ ] Find formal citations for new prompt axes (warmth, authority, guilt, transactional)
- [ ] M9 scoping should include post-training methodology docs for Qwen, DeepSeek, and Nemotron (not just HF lineage verification)
- [ ] Fix yaml.dump() API key mangling in update_config_model() (M1.T23, open since Apr 2)

---

## 2026-04-23 — Weekly Meeting with Prof. Introne: RQ Restructure, Paper Reframing

### Meeting Summary (Thursday 8:30am)

The first significant structural revision of the research questions since the lit review. The five-RQ structure drafted on Mar 24 was consolidated to three in the meeting transcript, then expanded back to four in the Apr 23 draft (architectural retained as RQ4 rather than collapsed into RQ2). The key conceptual move was reframing the paper from a model-performance benchmark into a practitioner decision resource that maps which constructs which models annotate well under which prompting strategy, with attribution layered on top. Prof. Introne explicitly advised against over-indexing on Rathje et al. 2024 as the primary framing paper — the contribution is no longer "GPT cannot capture X" but the multi-axis matrix.

### Revised Research Questions

- **RQ1 — Cultural variability from training corpus.** Descriptive plus main thesis. Absorbs the old "annotability hierarchy" question (which was dropped as a standalone RQ) by treating per-construct performance as the dependent variable in the corpus-culture comparison.
- **RQ2 — Prompt framing × cultural background.** Prof. Introne's favorite. Novel contribution. Layers culturally-grounded prompt axes (warmth, power/authority, guilt/emotional manipulation, businesslike/transactional) onto the 25 constructs and the model training culture dimension — directly operationalizing the Mar 26 "Matrix" insight. Absorbs the old binding-foundations mechanism question and the zero-shot vs few-shot methodology question as sub-hypotheses.
- **RQ3 — Training corpus vs post-training fine-tuning.** Conditional. Lives only if HuggingFace yields a clean model lineage with documented variants (RLHF v1/v2, DPO, Constitutional-AI variants). If not viable, fold into RQ1 and drop. Prof. Introne is comfortable with that outcome.
- **RQ4 — Architectural influence.** Size, quantization, MoE vs dense. Retained as the architectural axis — backup framing if RQ3 collapses, complementary if it survives.

### Reframing Decisions

Three specific moves came out of the discussion:

1. **Annotability hierarchy dropped** as a standalone RQ. The per-construct performance ranking is still in the data and still informs RQ1, but it is no longer framed as an independent question. This is consistent with Prof. Atari's Mar 18 directive: "LLMs can annotate" is old news, the contribution is in the variance.
2. **Zero-shot vs few-shot absorbed** into RQ2 as a sub-hypothesis. The 50-sample finding (0-shot beats multi-shot for all models) is now framed as one slice of the broader prompt-framing landscape, not a methodology question in its own right.
3. **Paper framing shifted** from a ground-truth claim to a practitioner-facing decision resource. The closing argument is no longer "this model is wrong about Purity" but "for this construct, in this cultural register, with this prompt framing, here is the model that performs best — and here is why."

### Outputs Produced

- `CAMEL_RQ_v2.docx` — three-RQ structure with preliminary hypotheses under each, "replaces or subsumes" annotations linking new questions to the original five, and a dropped/absorbed section explaining what happened to the old RQ1, RQ3, RQ4. (Architectural axis became RQ4 in subsequent revision.)
- `CAMEL_Literature_to_RQ_Mapping.docx` — all 25 reviewed papers organized into three tables by source group (prof-referenced, systematic-search, Tier 1), with primary bucket, secondary bucket, and role-in-argument columns. Bucket-level summary counts, plus a gaps section flagging the LLM-as-jury GLOBE-framework preprint mentioned in the meeting (still to be located) and noting that Wang et al. 2025 + Felix-Pena et al. 2025 (already in memory from Apr 2) belong in the RQ2 bucket.

### Submission Timeline Signal

Prof. Introne signaled the paper submission timeline should shift from mid-to-late April to early summer. Not formally committed, but the four-week buffer plus the OrangeGrid production runs ahead of us makes April unrealistic.

### Action Items

- [ ] Read Prof. Introne's belief dynamics paper as a close-reading exercise (academic writing craft)
- [ ] Locate LLM-as-jury GLOBE framework preprint
- [ ] Get formal RQ sign-off at next meeting (verbalized agreement was positive but not formal)
- [ ] Update WORKLOG / WORKPLAN / memory with the restructure (deferred to 4/30 session)

---

## 2026-04-03 — Weekly Meeting with Prof. Introne: Pre-Lit-Review-Followup Sync

### Meeting Summary

Routine weekly sync after the Mar 26 lit-review presentation and before the Apr 23 RQ restructure. Pre-meeting prep included: HPC pipeline cheat sheet (built Apr 2) covering the run_models.py for-loop, start_vllm/wait_for_vllm/kill_vllm flow, and HTCondor `condor_submit` workflow; an Apophis vs HPC comparison table across ten dimensions; and the literature-to-pipeline mapping document linking 15 papers to specific config2.yaml prompt IDs and Python function names. The cheat sheet was the centerpiece — Prof. Introne wanted to be able to explain the system end-to-end without me walking him through it line by line.

### Topics Covered

The meeting touched on the new prompt axes from Mar 26 (with Wang 2025 and Felix-Pena 2025 cited as the operant-conditioning framework justifying them), the training corpora composition research for the three validated models (Llama 3.3 ~95% English, Qwen 2.5 heavy Chinese, DeepSeek-R1-Distill dual Chinese), and the still-pending HPC cluster availability. The matrix concept from Mar 26 held — instrumental prompt axes × 25 constructs × model training culture — and the next step was supposed to be empirical: test the new axes on the 50-sample benchmark before committing to anything in the paper.

### Outstanding from This Session

The 50-sample test of the new prompt axes did not happen between Apr 3 and Apr 23 because the run_015 attempt on Apr 2 (with a config3.yaml that included annotation guide changes, not the new prompt axes) hit a yaml.dump() bug that mangled the API key on file rewrite, causing 401 Unauthorized errors on every model after Llama. The bug was diagnosed but not fixed, and the priority shifted to the lit review and RQ work.

### Action Items

- [ ] Test the four new prompt axes on the 50-sample (still outstanding)
- [ ] Fix yaml.dump() API key mangling in update_config_model()

---

## 2026-04-02 — Tooling Sprint, Config Debugging, Prompt Valence & Training Corpora Research

### fcat CLI Tool

Prof. Introne's offhand suggestion at the Mar 26 meeting was that a small Feather-file inspection utility would be useful given how often the pipeline produces `.feather` outputs. Built fcat as a Python hashbang script with subcommands: `head`, `tail`, `info`, `describe`, `columns`, `shape`, `cols`, `sample`, `unique`, `counts`, `query`, `grep`, `labels`, `export`. Deployed to `~/myenv/bin/fcat` so it's on PATH inside the venv. Also built `sanity_check.py` as a pre-flight script that confirms vLLM is up, the configured model is loaded, and the gold standard files exist before kicking off a full run.

### Config File Management

`config3.yaml` was created from `hpc/config3.yaml` to test annotation-guide tweaks (not the new prompt axes — that's a frequent point of confusion). `hpc/config2.yaml` was restored from base after some accidental edits. Column names in both HPC configs were corrected from `"Text Number"`/`"Text"` (50-sample Excel column names) to `"Number"`/`"text"` (56K corpus column names) using `sed`.

### YAML Apostrophe Bug

`config3.yaml` failed to parse on the first run with `expected <block end>, but found '<scalar>'`. Two unescaped apostrophes inside single-quoted strings — `group's inferiority` (line 367, in the Hate construct definition) and `object's actions` (line 491) — broke YAML's quoting rules. Fix: double the apostrophes (`group''s` / `object''s`). Both were patched with targeted `sed` and the file then validated cleanly with `python -c "import yaml; yaml.safe_load(open('config3.yaml'))"`.

### run_015_config3_test Issues

The overnight 50-sample run with `config3.yaml` hit two problems back-to-back. First, only 9 texts loaded instead of 50 — the config pointed at `samples/sample_prompts.xlsx` (the 9-text sample) rather than `samples/50_random_samples.xlsx`. Second, after Llama completed, every subsequent model returned 401 Unauthorized. Diagnosis: `yaml.dump()` inside `update_config_model()` is rewriting the YAML file between models and somehow mangling the `VLLM_API_KEY` value during the round-trip — possibly stripping or re-quoting characters. Not fixed before session end. Logged as M1.T23 in WORKPLAN; still open as of 4/30.

### Prompt Valence Literature Deep-Dive

Targeted hunt for sources to justify the new culturally-grounded prompt axes (warmth, power/authority, guilt, transactional) per Prof. Introne's Mar 26 directive that each prompt needs formal justification in the paper. Five additions to the working bibliography:

- **Wang et al. 2025 — "Behavioral Psychology of LLMs: Better Task Guidance Through Punishment and Reinforcement"** (Neurocomputing, July 2025). Strongest find. Introduces Behavioral Consequence Scenarios based on Skinner's operant conditioning, classifying prompts into positive reinforcement, negative reinforcement, positive punishment, negative punishment. Their BCSP method achieves 3.53/1.74 BLEU-2/4 improvement on dialogue. Provides the operant conditioning framework that makes the warmth/authority/guilt/transactional axes a coherent systematic exploration rather than four ad-hoc prompts.
- **Felix-Pena et al. 2025 — Prompt valence as behavioral control** (NeurIPS Workshop). Direct framing of prompt valence as a behavioral control mechanism. Pairs with Wang for the RQ2 citation set.
- **Yin et al. 2024 — "Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness"** (SICon @ EMNLP 2024). Politeness levels across English, Chinese, Japanese; impolite prompts hurt performance, overly polite doesn't help, and the optimal level differs by language. This is exactly the cultural-axis interaction Prof. Introne was pointing at — same prompt axis, different landings on Western vs. Eastern models.
- **Bulté & Rigouts Terryn 2025 — "LLMs and Cultural Values: The Impact of Prompt Language and Explicit Cultural Framing"** (Computational Linguistics, MIT Press, Dec 2025). Probed 10 LLMs with 63 items from Hofstede's VSM and WVS, translated into 11 languages, with and without cultural framing. Confirmed both prompt language and cultural perspective produce variation, but models stay anchored to a restricted set of countries (Netherlands, Germany, US, Japan) regardless of origin. Direct support for CAMEL's framing.
- **NegativePrompt attribution corrected.** Prior memory entry attributed NegativePrompt to "Li et al." — corrected to Wang et al. 2024. The earlier prompt-strategy literature reference was wrong.

### Training Corpora Composition Research

Per Prof. Introne's Mar 26 request, looked up what's actually known about each open-source model's training data:

- **Llama 3.3 70B (Meta):** ~15T tokens, ~95% English, US-aligned RLHF stack. Western-dominant by composition and alignment.
- **Qwen 2.5 72B (Alibaba):** ~18T tokens, heavy Chinese + English mix, post-trained at Alibaba in Hangzhou with Chinese-aligned safety/values training. Eastern-dominant on both axes.
- **DeepSeek-R1-Distill-Qwen-32B (DeepSeek):** Built on Qwen 2.5 32B base (14.8T tokens, Chinese+English), distilled with 800K reasoning samples generated by full DeepSeek-R1. Dual Chinese cultural encoding — base model is Chinese-aligned, distillation samples come from a Chinese-aligned reasoning model.

These distinctions sharpen the RQ1 framing: cultural orientation isn't binary, it's compositional, and DeepSeek is more "doubly Chinese" than Qwen by virtue of the distillation lineage. Worth a mention in the methods section.

### Documents Produced

Four Word documents to support the upcoming Apr 3 meeting and ongoing paper drafting:

- Detailed literature-to-pipeline mapping connecting 15 papers to specific `config2.yaml` prompt IDs and Python function names
- 2-page concise summary of the same mapping
- HPC pipeline cheat sheet with code references (the run_models.py for-loop, start_vllm/wait_for_vllm/kill_vllm flow, HTCondor `condor_submit` workflow)
- Apophis vs HPC comparison table across ten dimensions

### Self-Corrections During Session

Two corrections from Shahaan during the session worth noting for future fidelity: (a) Claude invented a run name `config3_warmth_test` when config3 only contained annotation-guide changes, not warmth prompts — the fictitious name conflated two different config experiments; (b) NegativePrompt was attributed to "Li et al." when the correct first author is Wang et al. Both noted in memory.

### Outstanding from This Session

- yaml.dump() API key mangling fix (M1.T23, still open)
- Test new prompt axes on 50-sample (deferred — config3 didn't have them yet)
- One-pager on paper thesis (later marked NOT NEEDED on 4/30 once the RQ structure replaced it)
- 25×9 per-label × per-model F1 table (later marked done — was in milestone doc Mar 12)

---

## 2026-03-26 — Weekly Meeting with Prof. Introne: Lit Review Presentation, "The Matrix", New Prompt Axes

*Narrative reconstructed 2026-07-12 from the WORKPLAN changelog; the original session left this entry as a header with no body.*

### Lit Review Presented

Presented the literature review draft to Prof. Introne — 25 papers across the two spreadsheets built March 24 (M6.T3 delivered). This was the deliverable committed at the March 19 meeting.

### Prompt Strategy Axis Expanded

The prompt-strategy dimension was expanded beyond the original negative-incentive framing. New culturally-grounded axes proposed: positive incentive / warmth, power / authority, guilt / emotional manipulation, and businesslike / transactional — not just negative incentive. Rationale is that power dynamics and emotional register land differently East vs West, so a single incentive axis undersells the variation.

### "The Matrix" — Key Insight

The central conceptual move of the meeting: instrumental prompt axes × 25 constructs × model training culture defines the novel contribution space. Distinguished ontological differences (what models get right/wrong about a construct) from processual differences (how prompting changes behavior). This "Matrix" framing later anchored the Apr 23 RQ2 restructure.

### HPC and Logistics

Prof. Introne offered Shahaan his own HPC account. HPC batch strategy discussed: disk-space constraints may require running model subsets sequentially and clearing weights between batches. A research TODO was set to investigate training-corpora cultural fingerprinting per model — this became M6.T8 (completed Apr 2).

---

## 2026-03-25 — Claude Code Setup, /DATA Migration, GitHub Cleanup & Push

### Claude Code Installation & Configuration

Installed Claude Code native binary (v2.1.79) on Apophis using Prof. Introne's
approval from the March 19 meeting. Global npm install (v2.1.17) already existed
but was outdated and required sudo for updates.

- Installed native binary via `claude install` → `~/.local/bin/claude`
- Added `~/.local/bin` to PATH in both `~/.bashrc` and `~/myenv/bin/activate`
  (venv activation was overriding PATH and hiding the native binary)
- Verified 2.1.79 works in both base shell and inside myenv
- Authenticated with personal Pro/Max subscription via OAuth
- Ran `/init` to generate `CLAUDE.md`, then replaced with corrected version
  including project guardrails, accurate environment info, and data locations
- Ran `/doctor` — clean diagnostics, native install confirmed

### /DATA Storage Migration

Executed the DATA_MIGRATION_PLAN.md to move data-heavy files from `/home`
to `/DATA/szkhan/camel/` on `/dev/sda1` (7.3TB, 2.9TB free).

**Completed:**
- Created directory structure: `/DATA/szkhan/camel/{samples,outputs,hpc_data/{chunks,results,logs}}`
- Moved `~/code_space3/samples/*` → `/DATA/szkhan/camel/samples/`
- Moved `~/code_space3/outputs/*` → `/DATA/szkhan/camel/outputs/`
- Moved `~/code_space3/hpc/chunks/*` → `/DATA/szkhan/camel/hpc_data/chunks/`
- Created symlinks from `~/code_space3` to `/DATA` for samples/, outputs/,
  hpc/chunks/, hpc/results/, hpc/logs/

**Note:** `hpc/results/` and `hpc/logs/` were empty (no files to move) — the
directories and symlinks are in place for when HPC runs begin in early April.

**Verified:** Pipeline files use relative paths, symlinks are transparent.
Config2.yaml reads samples/ and outputs/ without changes.

### GitHub Cleanup & Push

Executed the GITHUB_CLEANUP_PLAN.md — repo cleaned up and pushed.

**.gitignore updated:**
- Added `outputs/` (was commented out)
- Added `samples/CAMEL_cleaned_data_complete*.csv` (large corpus files)
- Added `hpc/chunks/`, `hpc/results/`, `hpc/logs/`, `hpc/*_wrapper.sh`, `hpc/*_submit.sub`
- Added `*.key`, `*api_key*`, `*_old.py`, `*_backup*`

**Files removed from git tracking (still on /DATA):**
- All output files (benchmarks, eval reports, run results) — ~80 files
- Large corpus CSVs (`CAMEL_cleaned_data_complete*.csv`)
- `hpc/merge_results.py` (merging done locally now)

**Files added:**
- `README.md` — project overview, pipeline architecture, quick start
- `test_data/` — small sample files preserved for cloning:
  `50_random_samples.csv`, `50_random_samples_ans.csv`,
  `sample_prompts.xlsx`, `sample_human_annotators_ans.xlsx`

**Files updated:**
- `hpc/HPC_CAMEL_README.md` — fixed Singularity version (v0.16.0), added conda
  vs venv note, real corpus paths, removed merge_results refs, added symlink
  info, fixed GPU routing (32B=1 GPU), fixed gold standard path

**Bash issue resolved:** Pasting markdown into terminal corrupted `.bashrc`
(wiped to 2 lines). Restored from `/etc/skel/.bashrc`, re-added PATH exports,
uncommented `force_color_prompt=yes`. Prompt colors restored.

### CLAUDE.md Guardrails

Replaced `/init`-generated CLAUDE.md with corrected version including:
- Project guardrails (never modify venv, vLLM, config without confirmation)
- Accurate environment info (~/myenv, Python 3.11.13, vLLM 0.16.0)
- Fixed CAMEL acronym (Expressions, not Expression Labeling)
- Data locations including planned /DATA migration
- Current model roster and research priority
- Pointers to WORKPLAN.md and WORKLOG.md

---

## 2026-03-24 — Literature Review Sprint: 25 Papers Compiled

### Lit Review Deep Dive (M6 Current Priority)

Conducted systematic literature review per Prof. Introne's March 19 directive
to become a subject matter expert. Produced two structured Excel spreadsheets
covering 25 total papers with full metadata, abstracts, methodology summaries,
key findings, and specific connections to CAMEL research.

### Spreadsheet 1: CAMEL_Literature_Review.xlsx (15 papers)

**Sheet 1 — Professor-Referenced Papers (5):**
1. Rathje et al. 2024 (PNAS) — GPT for multilingual psych text analysis.
   Enriched with full-text MFT results: Purity F1=0.144, Proportionality
   F1=0.130. Framed as "old news" baseline per Atari.
2. Kristensen-McLachlan et al. 2025 (PNAS Nexus) — Chatbot annotation reliability
3. Weber et al. 2018 (Comm Methods & Measures) — Moral content extraction difficulty
4. **CAMEL_V1 (Zewail, Setia, ... Atari, 2025)** — The corpus paper. 56,796 texts,
   30 annotators, 25 constructs. Our pipeline benchmarks LLMs against this.
5. **Atari et al. 2023 "Which Humans?"** — WEIRD hypothesis applied to LLMs.
   Co-authored by Atari + Henrich (Harvard). Theoretical backbone for East-West divergence.

**Sheet 2 — New Papers (10):**
1. Tao et al. 2024 (PNAS Nexus) — Cultural bias across 107 countries
2. Aksoy 2024 (AI and Ethics) — MFQ-2 across 8 languages, Arabic/Japanese deviate most
3. Wang et al. 2024 (ACL C3NLP) — CDEval: Qwen shows collectivism on Hofstede dimensions
4. Karinshak et al. 2024 (arXiv) — LLM-GLOBE: Chinese vs US model values
5. Abdurahman et al. 2024 (PNAS Nexus) — Perils of GPTology, from Atari's own lab
6. Bulla et al. 2025 (Elsevier) — Authority/Purity/Loyalty hard for ALL methods
7. Gilardi et al. 2023 (PNAS) — Foundational LLM-as-annotator paper (1,200+ citations)
8. Alizadeh et al. 2025 (J. Comp. Social Science) — Few-shot often hurts, validates 0-shot finding
9. Ziems et al. 2024 (Computational Linguistics) — 25 CSS benchmarks, task-dependent variation
10. Hopp et al. 2021 (Behavior Research Methods) — eMFD, pre-LLM baseline for MFT detection

### Spreadsheet 2: CAMEL_Lit_Review_Tier1.xlsx (10 papers)

Focused exclusively on Tier 1 venues (2023-2026) and major lab authors:
1. Santurkar et al. 2023 (ICML) — Stanford HAI: RLHF skews opinions
2. Hofmann et al. 2024 (Nature) — Allen AI/Stanford: RLHF masks covert bias
3. Scherrer et al. 2023 (NeurIPS Spotlight) — Columbia/Blei: moral hierarchy across 28 LLMs
4. Li et al. 2024 (NeurIPS) — Microsoft Research: CultureLLM, training data → values
5. Sclar et al. 2024 (ICLR) — UW/Allen AI/Yejin Choi: prompt sensitivity (76-pt swings)
6. Bail 2024 (PNAS) — Duke: field needs cross-cultural LLM benchmarks
7. Tan et al. 2024 (EMNLP Oral) — ASU/Huan Liu: definitive annotation survey
8. Abdulhai et al. 2024 (EMNLP) — Google DeepMind: MFT profiling of LLMs
9. Liu, Gurevych & Korhonen 2025 (TACL) — TU Darmstadt/Cambridge: cultural NLP taxonomy
10. Liu et al. 2024 (Findings of EMNLP) — Chinese vs Western moral divergence

### Key Themes Across the Literature

- **Training data geography drives cultural orientation** — confirmed by 6+ papers
  (Tao, Aksoy, Wang, Karinshak, Li/CultureLLM, Liu et al.)
- **Binding foundations (Authority, Loyalty, Purity) are universally harder** — confirmed
  by Bulla, Scherrer, Abdulhai, Rathje, Weber
- **0-shot ≥ few-shot for annotation** — supported by Sclar (formatting sensitivity),
  Alizadeh (explicit recommendation against few-shot), Rathje (mixed few-shot results)
- **RLHF/alignment can mask or amplify biases** — Santurkar, Hofmann, Scherrer
- **The field explicitly calls for what CAMEL does** — Bail, Liu/Gurevych/Korhonen, Tan

### Each spreadsheet has 17 columns per paper:
Citation, Year, Venue, Venue Type, Publisher Tier, Impact Factor, Est. Citations,
Author Credibility, Impact Level for CAMEL, Abstract, Methodology, Key Findings,
Conclusions, Connection to CAMEL (Brief), Connection to CAMEL (Full), Synthesis Notes.

### Five Research Questions Drafted

Based on the literature review findings and professors' framing:

1. **RQ1:** Which psychological constructs can LLMs reliably annotate, and which
   fall below acceptable agreement with human coders?
2. **RQ2:** How does the cultural origin of an LLM's training data systematically
   shape its annotation of moral and cultural constructs?
3. **RQ3:** Do binding moral foundations (Authority, Loyalty, Purity) represent a
   universal detection boundary for LLMs, or is detection failure culturally patterned?
4. **RQ4:** Why does zero-shot prompting consistently outperform few-shot prompting
   for psychological text annotation across model architectures?
5. **RQ5:** To what extent do RLHF and safety alignment procedures mask or amplify
   cultural biases in LLM text annotation?

Structure: RQ1 = descriptive finding, RQ2 = main thesis, RQ3 = mechanism,
RQ4 = methodology, RQ5 = connection to AI safety/alignment literature.

---

## 2026-03-19 — Weekly Meeting with Prof. Introne: Lit Review Pivot, HPC Hold, Storage & Repo Cleanup

### Meeting Summary (Thursday 8:30am)

HPC cluster is full — **2-week hold on all HPC testing** (target availability:
early April). Prof. Introne directed the next two weeks toward becoming a
subject matter expert via literature review, not implementation work.

### Key Directives

1. **Literature review is the priority.** Search arxiv.org for newest work.
   Focus on credible sources (Google DeepMind-tier venues, top conferences).
   Prof. acknowledged it's not reasonable to read everything — triage via
   abstracts and summaries, then deep-read only what's worth it.

2. **Analytical lens is critical.** Not just "LLMs are good at annotation" —
   the framing must be: "these LLMs are decent at *these* tasks and *these*
   variables, and this is *why*." Capturing the why adds context for both
   the paper and the coding/evaluation side.

3. **Next week deliverable:** Present a draft of lit review findings to
   Prof. Introne at the March 26 meeting. Whatever has been found so far.

4. **One-pager on paper thesis** (from Atari email) is deferred until after
   the lit review is substantive. Lit review informs the thesis, not the
   other way around.

5. **Model scope:** Keeping to the three for now (Llama, Qwen, DeepSeek).

6. **Paper legwork is on Shahaan.** Prof. Introne was not happy that Prof.
   Atari pushed the thesis framing back to us. The emails weren't as clear
   as Introne wanted. But it's now Shahaan's job to do the leg work and
   present findings — not the professors'.

### Storage Restructuring

Prof. reviewed disk usage on Apophis (`df -h`). Current `/home` partition
is 67% full (4.6TB of 7.3TB). Data-heavy outputs should move to `/DATA`
(`/dev/sda1`, 7.3TB total, 2.9TB free at 60%). Plan:

- Create `/DATA/szkhan/camel/` directory structure
- Symlink `samples/` and `outputs/` from `~/code_space3` to `/DATA`
- Keep code, configs, and small files under `/home`
- Option to zip large result sets to Google Drive for sharing
- Cold storage example shown: `/mnt/reddit-data` (62TB, Prof's Reddit data)
  — for reference only, not for our use

### Git / Repository Cleanup

- Add `.feather` files to `.gitignore` (prevents bloat from data files)
- Ensure no API keys anywhere in the repo
- Add READMEs to the root directory and each subdirectory
- Make the repo understandable and navigable for anyone reviewing it
- Remove `merge_results.py` from HPC folder — merging is done locally, not
  on HPC nodes
- Full corpus CSVs (`CAMEL_cleaned_data_complete.csv`,
  `CAMEL_cleaned_data_complete_ans.csv`) are in the repo but too large to
  view directly

### HPC Updates

- Prof. liked the GPU routing by model size and overall HPC setup
- Keep only the **100-text test chunk** for feather. Remove other chunk sizes.
  When HPC is ready, prof can run the test job with this chunk.
- Shahaan showed the system/user split in `prompt_builder.py` — prof
  understood the mechanism but full walkthrough of all updated files
  (adapter.py, run_annotation.py, evaluate.py) deferred to next time
- Test size locked at 100 texts for HPC validation

### Claude Code

Prof. Introne approved using **Claude Code directly on the Apophis server**.
This enables more interactive development and debugging on the remote machine.

### Action Items

- [ ] Deep-dive literature review (arxiv, top venues) — 2-week sprint
- [ ] Present lit review draft findings at March 26 meeting
- [ ] Move data to `/DATA/szkhan/camel/` with symlinks from `~/code_space3`
- [ ] GitHub cleanup: `.gitignore` update, READMEs, remove merge_results from HPC
- [ ] Keep only 100-text test chunk in HPC folder
- [ ] One-pager on thesis deferred until lit review is substantive

---

## 2026-03-18 — Prof. Atari Response, Prefix Caching, Feather I/O, HPC Batch Scripts

### Email Exchange: Atari → Introne → Atari (Mar 12–18)

Prof. Introne forwarded the milestone doc to Prof. Atari on Mar 12.
Four-email thread landed today with major strategic direction for the paper.

**Atari email 1 (Mar 17):** Liked the milestone. Two key directives:
1. Start with less expensive models before flagships — mentions GPT 5.2 mini
   getting good results from his students.
2. Paper framing must focus on psychological variables, not model performance.
   Requests a **one-pager on the main goal of the paper** before "filling in"
   results. Wants alignment on thesis before more runs.
3. Aside: familiar with Weber et al. 2018 paper on extracting latent moral
   information from text (René = René Weber). Side conversation with Introne,
   not directly relevant to our framing.

**Introne reply (Mar 18, 7:54am):** Agrees with variable-comparison framing.
Highlights three interesting dimensions:
1. Comparing across psychological variables × training corpora × architectures
2. Prompting effects — specifically Shahaan's incentive scheme making models
   more conservative, with possible "family differences" (model families)
3. Speculates about a "subtext on power dynamics and psychology" connecting
   prompt incentivization to model behavior — not yet theoretically grounded
   but worth exploring

Proposes either meeting to hash it out or iterating on a draft.

**Atari reply (Mar 18, 5:52pm):** Sharpens the thesis further:
1. "LLMs can annotate psych variables" is **old news** — cites Rathje et al.
   (PNAS 2024): GPT tested on 15 datasets, 47K texts, 12 languages for
   sentiment/emotions/offensiveness/moral foundations. Done.
2. **"GPT CANNOT capture some psych variables" might be the most interesting
   angle** — then compare across models, prompts, architectures.
3. Cites Kristensen-McLachlan et al. (PNAS Nexus 2025): "Are chatbots
   reliable text annotators? Sometimes" — systematic comparison of OS LLMs
   vs ChatGPT vs supervised ML. Directly relevant to our open-source focus.
4. Key framing: "text annotation is more than sentiment labeling and hate
   speech and incivility" — our corpus with 25 diverse psychological
   variables (cultural dimensions + moral foundations + behavioral markers)
   makes this possible.
5. Another angle for general audience: show LLMs are good (or sometimes
   slightly worse) than established simpler NLP tools.
6. Wants preliminary draft first, then meeting.

### Action Items from Email Thread

- [ ] **Draft one-pager on paper thesis/goal** (highest priority, before more runs)
  - Frame around: which psych variables can/can't LLMs capture, and why?
  - Position against Rathje et al. 2024 (old news) and Kristensen-McLachlan
    et al. 2025 (partial overlap, our contribution extends)
  - Three pillars: (1) variable annotability hierarchy, (2) cultural training
    effects, (3) prompt strategy × architecture interactions
  - Send to both profs, then schedule three-way meeting around it
- [ ] Prioritize cheaper API models (GPT 5.2 mini) before flagships
- [ ] Read Kristensen-McLachlan et al. 2025 in detail for positioning

### Key Reference Papers (from Atari)

1. Rathje et al. (2024). "GPT is an effective tool for multilingual
   psychological text analysis." PNAS, 121(34), e2308950121.
   — Establishes baseline: GPT can do psych annotation. Our paper goes beyond.

2. Kristensen-McLachlan et al. (2025). "Are chatbots reliable text
   annotators? Sometimes." PNAS Nexus, 4(4), pgaf069.
   — OS LLMs vs ChatGPT vs supervised ML. Closest to our work but limited
   to standard NLP tasks, not diverse psych constructs.

### vLLM Prefix Caching — Implemented & Tested

Restructured prompt delivery to use system/user message split. Static
prompt content (role, guidelines, label definition) sent as `system` message;
only the annotated text goes as `user` message. vLLM's automatic prefix
caching matches on the identical system prefix across all texts for the
same prompt × label.

**Files changed:** `adapter.py` (added `system_prompt` param to `call_model`
and `call_model_async`), `prompt_builder.py` (added `build_prompt_split()`),
`run_annotation.py` (uses split by default, `--no-split` flag for legacy).

**A/B benchmark results (9-text sample, 16 workers):**

| Model | Old RPM | New RPM | Change |
|-------|---------|---------|--------|
| Llama 3.3 70B | 246 | 263 | +7% |
| Qwen 2.5 72B | 158 | 172 | +9% |
| DeepSeek R1 32B | 88 | 87 | ~0% |

DeepSeek shows no improvement — bottleneck is output generation (reasoning
chains), not input processing. Gains compound at 56K scale: 1 cold compute +
56,795 cache hits per label.

No bleed-through: cached KV states encode only the static instruction prefix.
Each text gets a full forward pass. Predictions identical between modes.

### Feather Output Format — Implemented & Tested

Switched pipeline output from CSV to Apache Arrow Feather (`.feather`).
Matches format expected by `split_data.py` and HPC batch scripts.

**Files changed:** `run_annotation.py` (output now `.feather`, checkpoint
scans both `.feather` and `.csv`, flush changed from CSV append to Feather
overwrite every 50 rows), `evaluate.py` (auto-detects Feather/CSV on load,
eval reports save as `.feather`).

**New dependency:** `pyarrow` installed in `~/myenv`.

Backward compatible: checkpoint scanning reads both formats, old CSV runs
still work.

### HPC Batch Processing Scripts — Built & Ready for Testing

Adapted Prof. Introne's belief extraction HPC pipeline for CAMEL annotation.
All files in `~/code_space3/hpc/`.

**Files created:**
- `camel_annotate_hpc.py` — Self-contained annotation script. Starts vLLM
  via Singularity, loads Feather chunk, annotates all texts × prompts × labels
  with threading and system/user prefix caching, saves Feather results.
  `--no-server` flag for testing on Apophis without Singularity.
- `generate_batch_jobs.py` — Generates per-model HTCondor submit files with
  GPU routing by model size (70B → 2 GPUs, 32B/24B → 1 GPU).
- `wrapper.sh` — Template bash wrapper (per-model versions auto-generated).
- `test_job.sub` — HTCondor test submission for validation.
- `merge_results.py` — Combines chunk results into single Feather after jobs.
- `split_data.py` — Copied from project root (Prof. Introne's, unchanged).
- `config2.yaml` — Copied from project root, column names updated for 56K
  corpus (`Number`, `text`).

**GPU routing by model size:**

| Model | GPUs | TP | Node Pool |
|-------|------|----|-----------|
| Llama 3.3 70B | 2 | 2 | 2-GPU high-VRAM only |
| Qwen 2.5 72B | 2 | 2 | 2-GPU high-VRAM only |
| AceGPT v2 70B | 2 | 2 | 2-GPU high-VRAM only |
| DeepSeek R1 32B | 1 | 1 | Any node with 1 GPU |
| Mistral Small 24B | 1 | 1 | Any node with 1 GPU |
| Falcon-H1 34B | 1 | 1 | Any node with 1 GPU |

**Corpus prepared:** 56,796 texts converted to Feather, split into 12 chunks
of 5,000 texts each (+ 1 chunk of 1,796). Test chunk of 100 texts also
created. Answer key (with `Human Annotator Agreement` column) also converted.

**Ready to hand to Prof. Introne for HPC testing.**

---

## 2026-03-12 — Llama 3.3 Upgrade, Benchmarks, Doc Submitted to Prof. Introne

### Milestone Doc Submitted

Submitted `CAMEL_LLM_Annotation_Benchmarking_Milestone.docx` to Prof. Introne with:
- Full doc with all sections: annotation guidelines verbatim, label definitions
  + examples from codebook, 3 prompt templates, 50-sample results with Llama 3.3,
  25x9 F1 table, model selection roster, cost analysis with token estimates
- 3 raw result CSVs (one per model from run_007)
- eval_report CSV with "Keep the 0" fix applied
- Ollama vs vLLM timing report

Prof. Introne received positively. Forwarded to Prof. Atari same day.

### evaluate.py — "Keep the 0" Fix

Applied Prof. Introne's directive: zero-support labels now included in macro F1.
Removed `if metrics["support"] > 0` filter, updated macro precision/recall averaging.
Re-ran on run_007 results before submission.

### Llama 3.3 70B Upgrade

- vLLM model: `casperhansen/llama-3.3-70b-instruct-awq` (awq_marlin confirmed)
- Same VRAM, same TP=2 config as 3.1
- 9-text quick test (run_006), then 50-sample full run (run_007) completed

### Ollama vs vLLM Benchmark

Fair comparison: both backends at 16 threads, Ollama with OLLAMA_NUM_PARALLEL=16
+ OLLAMA_GPU_LAYERS=999 (full GPU offload).

| Model | Ollama RPM | vLLM RPM | Speedup |
|-------|-----------|---------|---------|
| Llama 3.3 70B | 44.3 | 195 | 4.4x |
| Qwen 2.5 72B | 28.8 | 162 | 5.6x |
| DeepSeek R1 32B | 18.6 | 93 | 5.0x |

Full corpus: Ollama 548 days vs vLLM 109 days = 439 days saved. Sent to prof as ammo.

---

## 2026-03-05 — 50-Sample Benchmark & Prof. Introne Meeting

### 50-Sample Benchmark Results (run_005_testing_50_r2)

Successfully ran all 3 models x 3 prompts against 50-text gold standard.
Results committed to GitHub in `outputs/run_005_testing_50_r2/`.

| Model | Best Prompt | F1-M | P-M | R-M | F1-m | Wall Time | RPM |
|-------|------------|------|-----|-----|------|-----------|-----|
| Llama 3.1 70B | incentive_strict | 0.285 | 0.250 | 0.407 | 0.333 | 1152s | 195 |
| DeepSeek R1 32B | 0-shot_binary | 0.245 | 0.269 | 0.325 | 0.275 | 2428s | 93 |
| Qwen 2.5 72B | 0-shot_binary | 0.196 | 0.248 | 0.179 | 0.333 | 1392s | 162 |

Full corpus ETA (1.42M requests, best prompt only):
Llama ~5 days, Qwen ~6 days, DeepSeek ~10.6 days

### Key Findings at 50-Sample Scale

1. **Llama best overall** (F1=0.285). 0-shot beats multi-shot across all models.
2. **East-West divergence confirmed**: Qwen systematically conservative with
   near-zero recall on Binding labels (Authority, Loyalty, Purity). Not a
   pipeline bug — replicates across all 3 prompts.
3. **DeepSeek trades precision for recall**: reasoning chains introduce
   over-interpretation (e.g., Honesty P=0.179, R=0.714).
4. **Label tiers**:
   - Reliable: Care, Ownership, OffensiveUncivilLanguage, Hate
   - Hard: CreativityInnovation, IntellectualHumility, AnalyticalThinking,
     GeneralizedTrust
   - Zero gold positives in 50-sample: Collectivism, Tightness, Looseness,
     Loyalty, Purity, Fear

### Meeting with Prof. Introne (Thurs 8:30am)

**Action items assigned:**

1. **evaluate.py — include zero-support labels in macro F1**
   - Prof: "Keep the 0, don't exclude it"
   - F1=0 for zero-support labels is a real signal, not undefined
   - Change: remove the `if support > 0` filter in macro averaging

2. **Per-label x per-model breakdown table**
   - 25 label rows x 9 columns (3 models x 3 prompts)
   - Row headers: label name + support count e.g. `Care (4)`
   - Bold the best F1 per row/column
   - Fit on 1 page — quick visual indicator of model strengths

3. **Prompt templates section in doc**
   - Add near top of meeting doc for Prof. Atari review
   - Show all 3 prompt templates with representative codebook examples

4. **HPC infrastructure — Ollama fallback + HTCondor**
   - CUDA drivers not updated on A1 nodes, only 12 machines for vLLM
   - Test Ollama benchmark speed vs vLLM → ammo for driver upgrade case
   - Build HTCondor batch submit: Feather input, 250K batches, route by model size
   - Small models (≤24B, ~12GB INT4) → abundant small nodes
   - 70B models (30-40GB INT4) → limited high-VRAM nodes

5. **Prompt caching / context prefill**
   - Cache static rubric + annotation guidelines as prefilled context
   - Only inject the text per request
   - Should dramatically boost throughput for all models

6. **7-model roster — pending sign-off**
   - Prof reviewing after doc cleanup
   - Three-way meeting with Prof. Atari planned to discuss prompt strategy
   - Suggested budget ask: $500 (Tier 1 x 3 models) to $2,500 (Tier 1 + flagships)

### LLM Landscape Research (completed pre-meeting)

- Full market analysis artifact created covering:
  - Frontier API models with pricing (GPT-5 family, Claude 4.6, Gemini 3.1,
    DeepSeek V3.2, Kimi K2.5)
  - Middle Eastern LLMs: Falcon-H1 Arabic (TII, UAE, leads OALL),
    Jais 2 70B (Inception/G42, largest open Arabic-first),
    ALLaM 34B (SDAIA, closed access), AceGPT v2 (KAUST, NeurIPS 2024),
    Fanar 2.0 (QCRI, Islamic RAG)
  - Citable 2026 leaderboard sources: Onyx, BentoML, Awesome Agents,
    Artificial Analysis, Lambda, Arena (LMSYS)
  - 5 directly relevant academic papers on cultural bias in LLM annotation

---

## 2026-03-04 — Token Overflow Fix, Run Organization & Checkpoint Design

### Input Token Overflow

- Prompts exceeded 2048 context window when combined with max_tokens=1024.
  Math: guidelines (~400) + definition (100-300) + examples (~400) + text
  + instructions = 1200+ input. 1200 + 1024 = 2224 > 2048.
- Fix: max_tokens=256 for Llama/Qwen (YES/NO = ~80 tokens), 1024 for
  DeepSeek (think chain IS output tokens, needs budget).

### Run Directory Organization

- Added `--run-name` flag and auto-incrementing `run_NNN` directories
- `get_next_run_number()` scans outputs/ for existing dirs
- `update_config_model()` now sets output_dir per run in config2.yaml
- Legacy files moved to `outputs/run_000_legacy/`

### Checkpoint/Resume

- Added `import glob` and skip_existing scan to run_annotation.py
- On resume with same `--run-name`, loads completed row keys from
  existing CSVs and skips them

---

## 2026-03-03 — Evaluation Module & Architecture Findings

### Evaluation Module (evaluate.py)

- Standalone scorer: F1-macro/micro, precision, recall per label/model/prompt
- Supports strict (majority) and lenient (any annotator) gold approaches
- Auto-saves CSV report, handles -1 predictions, filters Ollama errors

### Gold Standard

- `sample_human_annotators_ans.xlsx`: 9 texts, 34 annotator pool, 25 labels
- CAMEL V1 majority voting: 2+ of 3 annotators must agree
- Base rate: 41/225 positive (18.2%)

### DeepSeek-R1 Architecture Finding (Axis 2)

- Reasoning model wraps output in `<think>...</think>` before YES/NO
- Baked into weights, not promptable. Parser returned -1 for most predictions.
- Fix: regex strip `<think>.*?</think>` in parse_yes_no
- Paper relevance: reasoning-distilled models alter pipeline assumptions

### Qwen Ultra-Conservative Behavior (Axis 1)

- F1=0.154 is NOT a bug — parses correctly, genuinely says NO to everything
- Very low recall (0.064-0.127), moderate precision
- Legitimate East-West divergence finding

---

## 2026-03-02 — CUDA Fix & First Successful Pipeline Run

### CUDA Toolkit / nvcc Fix

- FlashInfer backend requires nvcc for JIT compilation
- Found at `/usr/local/cuda-12.2`, added to .bashrc and start_vllm()
- vLLM starts in ~80s with cached FlashInfer kernels

### Pipeline Debugging (3 failures before success)

1. Ollama model left enabled → disabled
2. max_tokens (4096) > max_model_len (2048) → lowered to 1024
3. Missing model entries in config2.yaml → ran update_config_models.py

### First Successful Automated Run

- run_models.py completed all 3 models on 9-text sample

---

## 2026-03-01 — Environment Overhaul & Pipeline Fixes

### Python 3.8 → 3.11 Upgrade

- Created ~/myenv with Python 3.11.13, vLLM 0.16.0, torch 2.9.1
- Fixed shebangs after venv rename, cosmetic prompt name

### Jupyter Lab Persistent Setup

- nohup on port 8891, token-only auth, multi-panel workspace

### run_models.py Fixes

- VLLM_ENV/PIPELINE_ENV → ~/myenv, removed --num-scheduler-steps,
  lowered gpu_memory_utilization to 0.90

---

## Pre-March — Pipeline Architecture (Previous Work)

### Modular Pipeline

- Migrated from camel_annotation_demo.py to config-driven architecture
- Files: config2.yaml, prompt_builder.py, adapter.py, run_annotation.py,
  run_models.py, benchmark_workers.py, update_config_models.py, evaluate.py
- Binary per-label YES/NO replaced multi-label-in-one-prompt approach
- Definitions verbatim from CAMEL Annotation Guide PDF

### Throughput Benchmarks

- Llama 3.1 70B: 165→195 RPM with 16 workers
- Qwen 2.5 72B: 162 RPM
- DeepSeek R1 32B: 93 RPM (slower due to think chains)