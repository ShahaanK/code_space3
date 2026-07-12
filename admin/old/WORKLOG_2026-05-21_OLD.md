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
