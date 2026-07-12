# WORKPLAN.md — CAMEL Cultural & Moral Bias in LLM Annotation

**Author:** Shahaan Khan
**Advisor:** Prof. Joshua Introne, Syracuse University iSchool
**Collaborator:** Prof. Mohammad Atari, UMass Amherst
**Target:** Open-ended (mid-to-late April target slipped; new date pending Introne discussion)

## Active Plan

### Milestone 1: Infrastructure & Environment
- [x] M1.T1 — Set up Apophis server access (SSH, RDS)
- [x] M1.T2 — Create Python environment with vLLM, torch, pipeline deps
- [x] M1.T3 — Upgrade Python 3.8 → 3.11 for vLLM 0.16.0 compatibility
- [x] M1.T4 — Set up persistent Jupyter Lab workspace (nohup, multi-panel)
- [x] M1.T5 — Fix vLLM 0.16.0 compatibility (deprecated flags, memory, CUDA_HOME)
- [x] M1.T6 — Fix FlashInfer JIT compilation (nvcc path via CUDA_HOME)
- [x] M1.T7 — Fix run_models.py subprocess env (CUDA_HOME in start_vllm())
- [x] M1.T8 — Sync config2.yaml model entries with run_models.py LOCAL_MODELS
- [x] M1.T9 — Disable stale Ollama model, fix max_tokens > max_model_len
- [ ] M1.T10 — Generate Jupyter config for persistence settings
- [x] M1.T11 — Implement checkpoint/resume for mid-run interruptions
- [ ] M1.T12 — Address label-order confound (randomize label order per text)
- [x] M1.T13 — Implement prompt caching / context prefill for throughput boost
- [ ] M1.T14 — Ollama benchmark test on HPC nodes (ammo for CUDA driver upgrade case)
- [x] M1.T15 — Migrate data to `/DATA/szkhan/camel/` (symlink samples/ and outputs/ from ~/code_space3)
- [x] M1.T16 — GitHub cleanup: .gitignore updated, README.md added, test_data/ preserved, outputs removed from tracking, pushed to GitHub
- [x] M1.T17 — Remove merge_results.py from hpc/ (merging done locally, not on HPC)
- [x] M1.T18 — Keep only 100-text test chunk in hpc/chunks/, remove other chunk sizes
- [x] M1.T19 — Claude Code approved for use on Apophis server
- [x] M1.T20 — Claude Code installed (native v2.1.79), authenticated, CLAUDE.md with guardrails deployed
- [x] M1.T21 — Build fcat CLI tool (Feather file reader: head/tail/info/describe/columns/shape/cols/sample/unique/counts/query/grep/labels/export), deployed to ~/myenv/bin/fcat
- [x] M1.T22 — Build sanity_check.py for pipeline readiness verification
- [ ] M1.T23 — Fix yaml.dump() in update_config_model() mangling API key on file rewrite (caused 401 on run_015 after Llama)

### Milestone 2: Pipeline Architecture
- [x] M2.T1 — Migrate from monolithic demo script to modular config-driven pipeline
- [x] M2.T2 — Build config2.yaml with verbatim CAMEL definitions and examples
- [x] M2.T3 — Implement binary per-label evaluation (YES/NO per label per text)
- [x] M2.T4 — Build adapter.py (OpenAI-compatible client for any provider)
- [x] M2.T5 — Build prompt_builder.py (conditional markers/examples assembly)
- [x] M2.T6 — Build run_annotation.py (sync/async/threaded modes)
- [x] M2.T7 — Build run_models.py (automated multi-model vLLM orchestrator)
- [x] M2.T8 — Build benchmark_workers.py (worker count throughput sweep)
- [x] M2.T9 — Build update_config_models.py (config sync utility)
- [x] M2.T10 — Fix parse_yes_no for DeepSeek-R1 <think> tag stripping
- [x] M2.T11 — Per-model max_tokens (256 for Llama/Qwen, 1024 for DeepSeek)
- [x] M2.T12 — Run directory organization (outputs/run_NNN_name/)
- [x] M2.T13 — Fix evaluate.py: include zero-support labels in macro F1 (prof: "Keep the 0")
- [x] M2.T14 — Build per-label x per-model F1 table (25 rows x 9 cols, bold winners, 1 page) — included in milestone doc submitted to Prof. Atari
- [x] M2.T15 — Switch pipeline output from CSV to Feather format for HPC scale
- [x] M2.T16 — Implement system/user prompt split for vLLM prefix caching
- [x] M2.T17 — Add --no-split flag for legacy single-message mode
- [x] M2.T18 — Fix YAML apostrophe escaping in config3.yaml (group's, object's broke parser)

### Milestone 3: Sample Evaluation
- [x] M3.T1 — Obtain gold standard annotations (9-text and 50-text samples)
- [x] M3.T2 — Understand CAMEL V1 aggregation method (majority voting, tri-value)
- [x] M3.T3 — Binarize gold standard (strict: majority only = positive)
- [x] M3.T4 — Build evaluate.py (F1-macro/micro, precision, recall per label/model/prompt)
- [x] M3.T5 — Run 3-model benchmark x 3 prompts x 9 texts
- [x] M3.T6 — First 9-text eval (Llama F1=0.335, Qwen F1=0.154, DeepSeek F1=0.074)
- [x] M3.T7 — Re-run with DeepSeek parse fix + token overflow fix
- [x] M3.T8 — 50-sample benchmark completed (run_005_testing_50_r2)
- [x] M3.T9 — 50-sample eval: Llama F1=0.285, DeepSeek F1=0.245, Qwen F1=0.196
- [x] M3.T10 — East-West divergence confirmed at 50-sample scale
- [x] M3.T11 — DeepSeek over-interpretation documented (Honesty P=0.179, R=0.714)
- [x] M3.T12 — Build 25x9 per-label x per-model F1 table for Prof. Introne — included in milestone doc
- [x] M3.T13 — Add prompt template examples to meeting doc (from codebook) — included in milestone doc
- [x] M3.T14 — 50-sample run with Llama 3.3 70B completed (run_007)
- [x] M3.T15 — Re-run evaluate.py on run_007 outputs with "Keep the 0" fix

### Milestone 4: Apophis Dev/Test Pipeline
**Note (4/30):** Originally framed as "HPC batch processing" assuming Apophis was the cluster. Clarified 4/30: Apophis is Prof. Introne's lab box (not Condor); OrangeGrid is the actual HPC cluster (see M8). M4 now scoped to Apophis dev/test only.

- [x] M4.T1 — Build HTCondor batch submit script (Feather input, per-model submit files)
- [x] M4.T2 — Route by model size: 70B → 2 GPUs, 32B/24B → 1 GPU
- [x] M4.T3 — Test Ollama on HPC nodes, benchmark timing vs vLLM (4.4–5.6x vLLM speedup, 439 days saved)
- [x] M4.T4 — Implement prompt caching / prefill (system/user split, tested +7–9% RPM)
- [x] M4.T5 — Convert 56K corpus to Feather, split into 12 chunks of 5,000 texts
- [x] M4.T6 — Build camel_annotate_hpc.py (self-contained: starts vLLM via Singularity, annotates chunk, saves Feather)
- [x] M4.T7 — Build generate_batch_jobs.py (per-model submit files with GPU routing)
- [x] M4.T8 — ~~Build merge_results.py~~ — completed but removed from repo; merging done locally on Apophis, not on HPC
- [x] M4.T9 — Create test chunk (100 texts) for HPC validation
- [x] M4.T10 — Apophis 70B AWQ pipeline VALIDATED — Llama 3.3, Qwen 2.5, DeepSeek-R1 all ran successfully on 2× A6000 (TP=2, INT4 AWQ, vLLM 0.16.0). RPMs: Llama 195, Qwen 162, DeepSeek 93.
- [ ] M4.T11 — Validate Apophis test_chunk job (100 texts → Feather output) end-to-end with current generate_batch_jobs.py + wrapper.sh

### Milestone 5: API Model Runs ⏸️ BLOCKED ON BUDGET APPROVAL
**Status:** Blocked since Mar 18 (~6 weeks). Awaiting Atari/Introne sign-off on $500–$2,500 cloud spend.

- [x] M5.T1 — Cost estimation completed (see cost table below — flagged for refresh)
- [ ] M5.T2 — Get budget approval from Prof. Introne / Prof. Atari ($500-$2,500) ⏸️ BLOCKED
- [ ] M5.T3 — Run GPT-5 Mini via API — Atari says students got good results with 5.2 mini
- [ ] M5.T4 — Run GPT-5 Nano via API
- [ ] M5.T5 — Run DeepSeek V3.2 via API (cheapest tier)
- [ ] M5.T6 — Run Claude Sonnet 4.6 via Batch API — only if budget approved
- [ ] M5.T7 — Evaluate cloud models against same gold standard
- [ ] M5.T8 — Compare LLMs against simpler NLP baselines (Atari: show LLMs vs established tools)

### Milestone 6: Literature Review & Subject Matter Expertise (Ongoing)
- [x] M6.T1 — Systematic arxiv search for LLM annotation / cultural bias / moral foundations
- [x] M6.T2 — Build annotated bibliography (2 Excel spreadsheets, 25 papers, 17 columns)
- [x] M6.T3 — Present lit review draft to Prof. Introne — DONE March 26 meeting
- [x] M6.T4 — Read Rathje et al. 2024 (PNAS) in detail — Purity F1=0.144, "old news" baseline
- [ ] M6.T5 — Read Kristensen-McLachlan et al. 2025 (PNAS Nexus) in detail — closest competitor
- [x] M6.T6 — Identify gaps in existing work that CAMEL fills (5 cross-cutting themes)
- [x] M6.T7 — Prompt valence/incentive literature deep-dive (Apr 2): Wang et al. 2025 (Neurocomputing, operant conditioning), Felix-Pena et al. 2025 (NeurIPS Workshop, prompt valence), Yin et al. 2024 (SICon@EMNLP, cross-lingual politeness), Bulté & Rigouts Terryn 2025 (CL, cultural framing)
- [x] M6.T8 — Training corpora composition research (Apr 2): Llama 3.3 ~95% English/15T tokens/US-RLHF; Qwen 2.5 18T tokens/heavy Chinese+English/Alibaba-aligned; DeepSeek-R1-Distill-Qwen 14.8T base + 800K distillation/dual Chinese cultural encoding
- [ ] M6.T9 — Locate LLM-as-jury GLOBE framework preprint (mentioned Apr 23 meeting, distinct from Karinshak)
- [ ] M6.T10 — Targeted Jinfen-aligned architecture literature search (for RQ4 backup framing — do not cite Jinfen's dissertation directly)
- [ ] M6.T11 — Find formal citations for new prompt axes (warmth, power/authority, guilt, transactional) — Prof. Introne wants justification per axis in paper

### Milestone 7: Paper Writing & Submission
**Status:** Deadline open-ended. Original mid-to-late April target slipped. New target pending Introne discussion (signaled "early summer" at Apr 23 meeting).

- [x] M7.T1 — Milestone doc submitted to Prof. Introne, forwarded to Prof. Atari (received positively)
- [x] M7.T1b — Initial five research questions drafted from lit review findings
- [x] M7.T1c — RQ structure REVISED Apr 23: consolidated 5 → 4 RQs (3 in transcript + architectural retained as RQ4)
  - RQ1: Cultural variability from training corpus (descriptive + main thesis)
  - RQ2: Prompt framing × cultural background — culturally-grounded axes (warmth, power/authority, guilt, transactional) layered on 25 constructs and model training culture (Mar 26 "Matrix"). Prof's favorite. Absorbs binding mechanism + 0-shot vs few-shot.
  - RQ3: Training corpus vs post-training fine-tuning — CONDITIONAL on finding clean HF model lineages with documented variants
  - RQ4: Architectural influence (size, quantization, MoE vs dense)
- [x] M7.T1d — CAMEL_RQ_v2.docx + CAMEL_Literature_to_RQ_Mapping.docx produced (Apr 23)
- [ ] M7.T1e — RQ2 culturally-grounded prompt axes (warmth, power/authority, guilt, transactional) implementation DEFERRED to post-base-wave. First 56K production runs use the existing 4 prompting strategies (zero-shot, few-shot, minimal, incentive_strict). New axes added as a second wave once the base baseline is locked. (Decision: 5/14 meeting, Prof. Introne agreed)
- [ ] M7.T2 — ~~One-pager on paper thesis~~ — NO LONGER NEEDED (RQ structure now serves this purpose)
- [ ] M7.T3 — Three-way meeting with Prof. Introne + Prof. Atari (pending RQ sign-off + Atari budget)
- [ ] M7.T4 — Write methods section (pipeline, evaluation, model selection)
- [ ] M7.T5 — Write results: East-West divergence, architecture effects, label tiers
- [ ] M7.T6 — Justify model selection via 2026 leaderboards (HF Open LLM Leaderboard v2, OALL v2)
- [ ] M7.T7 — Final draft and submission — DEADLINE OPEN-ENDED (April target slipped)
- [ ] M7.T8 — Get Prof. Introne's formal sign-off on 4-RQ structure (verbalized agreement at Apr 23 was positive but not formal)

### Milestone 8: OrangeGrid HPC Setup & Production Runs ⚡ (post-4/30)
**Status (2026-07-06):** Pipeline stood up and validated on OrangeGrid. V1 engine unlocked (224 RPM, beats Apophis). Llama + Qwen 100-text characterization complete. **Production 56K wave gated** on: (a) the DeepSeek deterministic-annotation failure (M8.T10h), and (b) the F1 regression diagnosis (M8.T10i). Llama and Qwen are unblocked and can proceed once the F1 question is settled.

- [x] M8.T1 — Request OrangeGrid access (approved 4/29 by Dan/ITS)
- [x] M8.T2 — First SSH connection to its-og-login5.syr.edu via SU RDS (5/7)
- [x] M8.T3 — Empirical reconnaissance (5/7 + 5/11): GPU pool inventory, ClassAd attribute discovery (Condor 9.8 uses CUDA-prefix attrs), default request_disk, PREEMPT=False, Singularity 3.7.1 / Apptainer 1.1.3
- [x] M8.T4 — ITS A100 access query resolved: A100s made visible to szkhan account (9 nodes / 18 A100 80GB GPUs, driver 12.8)
- [x] M8.T5 — Set up OrangeGrid environment (Miniforge 24.7.1 at /home/szkhan/miniconda3, camel conda env Python 3.11, deps installed) (5/19)
- [x] M8.T6 — vLLM SIF on OrangeGrid (Prof. Introne SCP'd vllm-openai.sif; copied 5/19; vLLM 0.10.1.dev1+gbcc0a3cbe / commit bcc0a3c verified)
- [x] M8.T7 — Pre-cache HuggingFace model weights to /home/szkhan (Llama 3.3 70B AWQ, Qwen 2.5 72B AWQ, DeepSeek-R1 32B AWQ; 95GB total) (5/21)
- [x] M8.T8 — Clone code from GitHub to OrangeGrid via PAT; rsync chunks/ from Apophis (5/21)
- [x] M8.T8b — Claude Code SIF deep-dive (5/21): awq_marlin kernel confirmed at runtime, python3 (not python) required inside container, V0 forced by fp8 KV cache flag
- [x] M8.T9 — Adapt batch scripts for OrangeGrid (test_job_og.sub + wrapper.sh, 12 gotcha fixes; TP=1, A100 80GB via CUDADeviceName ClassAd, HF cache bind) (5/21)
- [x] M8.T10 — Submit OrangeGrid 100-text test (Llama 3.3 70B); job 469622 clean exit, 7,500 API calls, 300-row Feather; steady-state 63 RPM (V0 baseline) (5/21–5/28)
- [ ] M8.T10b — 100-text Llama with system/user prompt split; measure prefix-caching RPM delta on vLLM 0.10.1.dev (DEFERRED behind V1 decision)
- [x] M8.T10c — V1 engine experiment: removed --kv-cache-dtype fp8_e4m3, unlocking V1; Llama 3.3 63 → 224 RPM (~3.6×, beats Apophis 195), annotation-neutral. Overturns the 5/21 "3× slower" conclusion (6/5)
- [ ] M8.T10d — Apophis re-baseline (verify 195 RPM is API-call rate, not row rate)
- [ ] M8.T10e — config2 vs config3 prompt-token impact (re-run test on config2, compare RPM + F1)
- [ ] M8.T10f — Concurrent thread scaling (16 vs 32 vs 64 threads)
- [x] M8.T10g — Llama 3.3 + Qwen 2.5 100-text characterization complete under V1 (6/5)
- [ ] M8.T10h — ❌ DeepSeek-R1 deterministic annotation FAILING. Bare greedy → repetition collapse. repetition_penalty 1.15 → 36.5% unparseable (penalty-induced incoherence) on 100-text (7/6). Second deterministic config to fail. Next: try repetition_penalty ~1.05; else Option 4 (document as reasoning-model exception, or swap Eastern reasoning slot for Qwen3 non-thinking — Introne/Atari decision). Full corpus gated for DeepSeek only
- [ ] M8.T10i — F1 regression investigation: 100-text incentive_strict macro F1 = 0.208 vs March 0.285 (~27% drop). Hypotheses: config3 longer definitions; Llama 3.1→3.3 not directly comparable; 50 vs 100 sample variance. Isolate config2 vs config3 on Llama before burning A100 time. **Gates the production wave**
- [ ] M8.T11 — 1,000-text characterization wave (3 models in parallel) — or chunk at 1,000 texts/chunk (57 chunks) for production per the 7/6 plan
- [ ] M8.T12 — Submit full 56K production runs (subset protocol per Introne: first chunk per model, verify, then release the rest); Llama + Qwen first, DeepSeek gated on M8.T10h
- [ ] M8.T13 — Merge chunk results locally on Apophis (NOT on OrangeGrid); evaluate at scale
- [ ] M8.T14 — Statistical significance tests for East-West divergence at full corpus scale
- [x] M8.T15 — First per-chunk timing report to Introne (Production_Runtime_Estimates_Rev2.docx)
- [ ] M8.T16 — Ongoing coordination pings with Introne during the production wave (he has CPU clustering + GPU work queued; pause requests on >30GB-VRAM jobs)
- [ ] M8.T17 — Regenerate OG GitHub PAT with Contents: Read+Write to end OG→GitHub drift (currently read-only; hpc/ edits must be promoted OG→Apophis and committed from Apophis)
- [ ] M8.T18 — Production_Runtime_Estimates_Rev3.docx with locked V1 numbers and Introne's 75%-of-Apophis baseline framing

### Milestone 9: RQ3 Viability Scoping ⚡ NEW (4-hour timebox)
**Status:** Conditional research task. RQ3 lives or dies based on this scoping.

- [ ] M9.T1 — Allen AI OLMo + Tülu lineage (most likely hit — fully documented SFT/DPO/RLHF variants)
- [ ] M9.T2 — Nvidia Nemotron family (Prof. Introne's suggestion — fully open training data)
- [ ] M9.T3 — Llama 3 base → Instruct → community fine-tunes (Tülu-Llama, Hermes — same base, varied methods)
- [ ] M9.T4 — Qwen base vs. Qwen-Instruct vs. Qwen-Chat (Alibaba post-training documentation)
- [ ] M9.T5 — Zephyr / HuggingFaceH4 (small but well-documented DPO vs SFT comparisons)
- [ ] M9.T6 — Decision: keep RQ3 with chosen lineage, or fold into RQ1 and drop. Report to Introne.

### Side Items
- [x] DGX Spark deliverable — local AI deployment comparison notes for the businessman conversation. Full landscape report drafted 4/30 (DGX Spark, Mac Studio, GPU workstations, cloud); resent to Prof. Introne 5/14 (he had not seen the earlier draft).
- [ ] CCDS proposal — ongoing Slack dialogue with Prof. Introne (private, internal proposal received and reviewed)
- [ ] Send Production_Runtime_Estimates to Introne — Rev2 pending send after final test numbers; supersede with Rev3 (M8.T18) once V1 numbers and the 75% baseline framing are locked
- [ ] NVIDIA deck — Use Case 1 slide rebuilt around cost economics 6/13 ("On-prem wasn't the cheap option, it was the only one"); ~$5,135 API vs $2,500 cap
- [ ] Cost summary table (below) — flagged for refresh; uses GPT-5 Mini / Claude 4.6 prices, current pricing is GPT-5.4-mini $0.25/$2.00 and Claude Opus 4.7

---

## Reference: Cost Summary (per full 1.42M-request run) — PENDING UPDATE

| Tier | Model | Total Cost |
|------|-------|-----------|
| Affordable | DeepSeek V3.2 (w/ cache) | ~$48 |
| Affordable | GPT-5 Nano | ~$75 |
| Affordable | GPT-5 Mini | ~$377 |
| Flagship | Claude Sonnet 4.6 (Batch API) | ~$2,133 |
| Flagship | GPT-5 | ~$1,884 |
| Prohibitive | Claude Opus 4.6 | ~$7,110 |

**Note:** Numbers stale as of 4/30. GPT-5.4-mini is now $0.25/$2.00 per million tokens; Claude Opus updated to 4.7. Refresh deferred until M5 budget approval discussion with Atari.

## Reference: Model Roster — Validated vs Planned (4/30)

| # | Model | Bucket | Origin | Size | Apophis Status | OrangeGrid Status |
|---|-------|--------|--------|------|----------------|-------------------|
| 1 | Llama 3.3 70B | West | Meta, USA | 70B | ✅ VALIDATED (TP=2, RPM=195*) | Pending M8 |
| 2 | Qwen 2.5 72B | East | Alibaba, China | 72B | ✅ VALIDATED (TP=2, RPM=162*) | Pending M8 |
| 3 | DeepSeek R1 32B | East | DeepSeek, China | 32B | ✅ VALIDATED (TP=1, RPM=93*) | Pending M8 |
| 4 | Mistral Small 3.1 24B | West | Mistral, France | 24B | Planned, not run | Future |
| 5 | AceGPT v2 70B | East | KAUST, Saudi Arabia | 70B | Planned, not run | Future |
| 6 | Falcon-H1 34B | East | TII, UAE | 34B | Planned, not run | Future |

*RPMs are benchmarks from March 50-sample runs, not hard rules.

**The north-star comparison is East vs West.** Specific country/org origin documented but the two buckets are the analytical axis.

**Note:** Cluster-specific GPU routing differs. Apophis: TP=2 for 70Bs across both A6000s. OrangeGrid: TP=1 with single A100 80GB per Prof's 4/23 guidance.

## Reference: HPC File Inventory (hpc/)

| File | Purpose |
|------|---------|
| `camel_annotate_hpc.py` | Self-contained annotation (starts vLLM, processes chunk, saves Feather) |
| `generate_batch_jobs.py` | Per-model HTCondor submit files with GPU routing |
| `split_data.py` | Splits corpus into Feather chunks |
| `test_job.sub` | HTCondor test submission (Apophis-style: TP=2) |
| `wrapper.sh` | Template wrapper (per-model versions auto-generated) |
| `config2.yaml` | Prompts, labels, guidelines (columns: Number, text) |
| `chunks/test_chunk.feather` | 100-text test chunk for validation |

**Removed:** `merge_results.py` — merging now done locally on Apophis post-run.

## Reference: Cluster Topology (current through 2026-07-06)

| | Apophis | OrangeGrid |
|---|---------|------------|
| Owner | Prof. Introne (iSchool lab) | Syracuse ITS |
| GPUs | 2× RTX A6000 (48 GB, sm_86) | Effective pool 27 nodes / 68 GPUs: 9× A100 80GB PCIe + 7× A40 + 11× L40S (via SUrge; RTX 6000/5000 excluded, CUDA 11.6) |
| Total VRAM | 96 GB on one box | Per-job, anywhere in pool |
| Scheduler | None (direct vLLM) | HTCondor 9.8.0 |
| vLLM version | 0.16.0 (pinned, do not touch) | 0.10.1.dev1+gbcc0a3cbe (commit bcc0a3c; capped by CUDA 12.8) |
| vLLM engine | V1 (default) | V1 (fp8 KV-cache flag removed 6/5; keep removed, V1 unavailable with that flag in 0.10.1.dev) |
| GPU config | 2× A6000, TP=2 | 1× A100 80GB, TP=1 |
| Llama 3.3 RPM | 195 (API-call rate pending re-verification, M8.T10d) | 63 at V0 → **224 at V1** (6/5, beats Apophis) |
| Storage | NFS home + /DATA/szkhan/camel | NetApp /home/szkhan, 4 TB, auto-mounted on compute |
| Login | apophis.ischool.syr.edu (SSH+RDS) | its-og-login5.syr.edu (SSH+SU RDS) |
| Status | Working, validated | Operational; pipeline validated, production wave gated (M8.T10h/T10i) |
| Role | Dev/test | Production 56K runs |

## Reference: Key Positioning Papers (from Prof. Atari, 2026-03-18)

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Rathje et al. 2024 — "GPT is an effective tool for multilingual psychological text analysis" | PNAS 121(34) | **Old news baseline.** GPT on 15 datasets, 47K texts, 12 languages, 4 constructs. Our paper must go beyond this. |
| Kristensen-McLachlan et al. 2025 — "Are chatbots reliable text annotators? Sometimes" | PNAS Nexus 4(4) | **Closest competitor.** OS LLMs vs ChatGPT vs supervised ML. Limited to standard NLP tasks, not diverse psych constructs. |
| Weber et al. 2018 — "Extracting latent moral information from text narratives" | Comm Methods & Measures 12(2-3) | **Methodological ancestor.** Crowd-truth coding of MFT foundations. |

## Reference: New Prompt Axes Literature (from Apr 2 deep-dive)

| Paper | Venue | Bucket |
|-------|-------|--------|
| Wang et al. 2025 — "Behavioral Psychology of LLMs: Better Task Guidance Through Punishment and Reinforcement" | Neurocomputing | Operant conditioning framework — justifies warmth/authority/guilt/transactional axes |
| Felix-Pena et al. 2025 — Prompt valence as behavioral control | NeurIPS Workshop | Prompt valence direct framing |
| Yin et al. 2024 — "Should We Respect LLMs?" | SICon @ EMNLP | Cross-lingual politeness; optimal level differs by language — supports cultural-axis framing |
| Bulté & Rigouts Terryn 2025 — "LLMs and Cultural Values: Prompt Language and Explicit Cultural Framing" | Computational Linguistics (MIT Press) | Cross-cultural prompt × cultural framing variation |

## Reference: Canonical OrangeGrid A100 Submit-File Recipe (confirmed in production 5/21)

Required ClassAds for jobs targeting NVIDIA A100 80GB on OrangeGrid (not documented in HPC_CAMEL_README.md):

```
request_gpus    = N
+request_gpus   = N
+wantsArgusNode = True
request_cpus    = 8
request_memory  = 64GB
should_transfer_files = NO
Requirements = (CUDADeviceName == "NVIDIA A100 80GB PCIe") && (TARGET.GPUs >= 2)
environment = "HOME=/home/szkhan CAMEL_TP=1 CAMEL_CONFIG=config3.yaml CAMEL_MODEL=<full HF path> HF_HOME=/home/szkhan/.cache/huggingface"
```

Critical: do NOT add `when_to_transfer_output = ON_EXIT` (contradicts ShouldTransferFiles=NO, errors at dry-run). `TARGET.GPUs >= 2` is a proxy for "both A100s on the node are free," avoiding GPU-memory contention. `CAMEL_MODEL` must be set or the template wrapper defaults to Llama (and skips the DeepSeek override).

**wrapper.sh required modifications vs the Apophis template:**
1. Accept a config override: `CONFIG_FILE="${CAMEL_CONFIG:-config2.yaml}"`
2. Convert CUDA_VISIBLE_DEVICES UUIDs → integer indices before invoking Python:
```bash
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
fi
```
3. Be executable: `chmod +x wrapper.sh` (git clone does not preserve the bit).

**Repository directory symlinks:** the Apophis repo has `chunks/`, `results/`, `logs/` as symlinks into `/DATA/szkhan/camel/hpc_data/...`; on OrangeGrid (no `/DATA`) they arrive broken and must be replaced with real dirs: `rm chunks results logs && mkdir chunks results logs`.

## Reference: Production Wall-Clock Estimates

Arithmetic correction (5/21): each (text × prompt) = **25 API calls** (one per label), not 1 request. A 100-text test = 100 × 3 × 25 = 7,500 calls.

| Scale | API calls | OG wall-clock at 63 RPM (V0) |
|---|---|---|
| 100 texts (test) | 7,500 | ~2 hours |
| 1,000 texts | 75,000 | ~19 hours |
| 5,000 texts (1 production chunk) | 375,000 | ~96 hours (4 days) |
| 56,796 texts (1 model, parallel chunks) | 4.26M | ~4 days |
| Full 3-model corpus (parallel) | 12.78M | 8 to 14 days (contention-dependent) |

**Note:** these are V0 (63 RPM) figures. The V1 unlock (224 RPM, 6/5) roughly quarters them; a Rev3 estimates doc with locked V1 numbers and Introne's "~75% of Apophis" baseline framing is pending (M8.T18). The sharp "3× slower than Apophis" framing from the 5/21 session has been superseded.

---

## Changelog

### 2026-07-06
- (Shahaan) ✅ DeepSeek repetition-penalty fix deployed to camel_annotate_hpc.py (code, not config): max_tokens 1024→2048, DEEPSEEK_REPETITION_PENALTY=1.15 in MODEL_OVERRIDES, conditional plumbing through call_vllm/annotate_chunk/main; Llama/Qwen payloads unchanged, temperature stays 0
- (Shahaan) ❌ M8.T10h — DeepSeek re-run FAILED at scale: 36.5% unparseable on 100-text. Penalty 1.15 traded repetition collapse for penalty-induced incoherence. Live single-curl looked clean; full run did not. OG log confirms penalty was applied. Second deterministic config to fail DeepSeek
- (Shahaan) 🔧 V1 near-miss: Apophis copy still had fp8 active (V0); nearly reverted V1 on OG via push. Caught via nvidia-smi (92% util) + grep; re-commented fp8 on OG, realigned both boxes. Rule: diff before promoting across boxes
- (Shahaan) 🔧 CAMEL_MODEL trap: wrapper.sh defaults to Llama unless CAMEL_MODEL set; added to test_job_og.sub environment. Also fixed nano-corrupted arguments line via sed
- (Shahaan) ✅ Version-control reconciliation: OG authoritative for hpc/ code; promoted config3 (apostrophe fix), wrapper.sh, submit files, condor_alert.sh OG→Apophis; committed from Apophis (SSH remote pushes; OG PAT read-only). All three aligned on hpc/ code
- (Shahaan) 🔧 CLAUDE.md refreshed (local to Apophis); VISION.md created; WORKLOG/WORKPLAN updated
- (Shahaan) 📋 DeepSeek decision pending: try repetition_penalty 1.05, else Option 4 (document exception / swap to Qwen3 non-thinking, Introne/Atari call). Full corpus gated for DeepSeek; Llama/Qwen unaffected
- (Shahaan) 🆕 M8.T17 — regenerate OG PAT with Read+Write; M8.T18 — Rev3 runtime estimates doc

### 2026-06-13
- (Shahaan) 📄 DeepSeek failure diagnosis doc for Introne (5pp): verbatim failure examples, per-construct parse-failure table, 4 ranked remediation options, all temperature-0 preserving
- (Shahaan) 🔧 Diagnosis: deterministic repetition penalty (repetition_penalty ~1.15 / frequency_penalty ~0.4), not raising max_tokens; degeneration visible in first 500 chars, not truncation. (Superseded 7/6: 1.15 induces a different degeneration at scale)
- (Shahaan) 📄 NVIDIA Use Case 1 slide rebuilt around cost economics ("On-prem wasn't the cheap option, it was the only one"); full benchmark ~$5,135 API vs $2,500 cap; 3 wrong framings corrected

### 2026-06-05
- (Shahaan) ✅ M8.T10c — V1 engine unlocked: removed fp8 KV cache flag, 63 → 224 RPM (3.6x, beats Apophis), annotation-neutral. Overturns 5/21 "3x slower" conclusion
- (Shahaan) ✅ M8.T10g — Llama 3.3 + Qwen 2.5 100-text characterization complete under V1
- (Shahaan) 🚫 M8.T10h — DeepSeek-R1 stalled on cluster contention after 2 resolved config failures; re-run deferred
- (Shahaan) 💡 Central finding: Llama over-predicts, Qwen under-predicts, near-identical aggregate F1; both collapse to ~0 F1 on moral foundations (Care, Equality, Loyalty, Authority, Proportionality, Purity)
- (Shahaan) 📄 Two faculty deliverables: 4-page deep dive + 1-page executive summary
- (Shahaan) 🆕 M8.T10i — F1 regression investigation (0.208 vs 0.285) added; gates production

### 2026-05-28
- (Shahaan) ✅ M8.T10 — Job 469622 outcome: clean exit, 7,500 calls, 300-row Feather. Steady-state locked at 63 RPM (V0 baseline)
- (Shahaan) 📋 100-text eval: incentive_strict macro F1 = 0.208 (best of 3 prompts); ~27% below March 0.285; shape consistent (incentive_strict > multi-shot ~ 0-shot)
- (Shahaan) 🔧 HPC pipeline writes to positional CLI output path, ignores config output_dir/prefix — by design for per-chunk collision avoidance
- (Shahaan) 🔧 Throughput reframe per 5/21 meeting: Introne's ~75% of Apophis is expected HPC overhead; soften "3x slowdown"; V1 experiment is the recoverable-throughput lever
- (Shahaan) 🔧 condor_history ages out ~1 week; rely on .out/.log. vLLM stateless per call, aggressively prefix-caches; hold-edit-release preferred over rm-resubmit
- (Shahaan) 🔧 V1 experiment staged (fp8 commented at camel_annotate_hpc.py:209, test_job_og.sub → test_results_v1_nofp8.feather); V0 artifacts preserved under _v0_fp8_run469622

### 2026-05-21
- (Shahaan) ✅ M8.T7 — HF model cache complete on OG (95GB, all 3 models)
- (Shahaan) ✅ M8.T8 — Repo cloned to OG via PAT (rsync rejected due to .venv-camel bulk)
- (Shahaan) ✅ M8.T8b — SIF deep-dive: vLLM 0.10.1.dev1+gbcc0a3cbe, awq_marlin at runtime, python3 required, V0 forced by fp8 flag
- (Shahaan) ✅ M8.T9 — Batch script adapted (test_job_og.sub, 12 gotcha fixes)
- (Shahaan) ✅ M8.T10 — 100-text Llama test run (job 469622); live rate 65 RPM (V0), finalized 63 at exit
- (Shahaan) 🔧 Arithmetic correction: each (text × prompt) = 25 API calls (one per label). 100-text test = 7,500 calls
- (Shahaan) 📄 Production_Runtime_Estimates_Rev2.docx produced with corrected math + live OG rate
- (Shahaan) 🆕 M8.T10c/d/e/f — V1 experiment, Apophis re-baseline, config2-vs-config3, thread scaling


### 2026-05-19
- (Shahaan) ✅ M8.T2: First SSH to OrangeGrid completed (already done 5/7; reconfirmed)
- (Shahaan) ✅ M8.T6: vLLM SIF on OrangeGrid (Prof. Introne provided via Apophis SCP; copied to /home/szkhan/vllm-openai.sif)
- (Shahaan) ✅ M8.T5: Miniforge 24.7.1 installed at /home/szkhan/miniconda3 per SU OrangeGridExamples canonical pattern
- (Shahaan) ✅ camel conda env created (Python 3.11) with pandas, pyarrow, pyyaml, huggingface_hub, openai, requests
- (Shahaan) ✅ vLLM 0.10.1.dev1+gbcc0a3cbe verified inside SIF (NOT pristine 0.10.0; commit bcc0a3c)
- (Shahaan) 🔧 SU canonical install path uses Miniforge not vanilla Miniconda; switched to that
- (Shahaan) 📋 Container invocation requires `python3` not `python`; documented in OrangeGrid Quickstart
- (Shahaan) 📋 `--nv` flag warning on login node is expected (no GPU drivers); harmless

### 2026-05-14
- (Shahaan) 📋 Weekly meeting with Prof. Introne (Thursday 8:30am)
- (Shahaan) 🔧 vLLM version split FORMALIZED: OrangeGrid pinned at 0.10.x (CUDA 12.8 max), Apophis stays at 0.16.0
- (Shahaan) ✅ M8.T6 unblocked: Prof. SCP'd his vllm-openai.sif to /DATA/ on Apophis during meeting
- (Shahaan) ✅ A100 access reconfirmed operational; single-A100 TP=1 strategy holds
- (Shahaan) 🆕 M8.T15: Submit production runs in subsets, characterize per-chunk runtime
- (Shahaan) 🆕 M8.T16: Coordinate GPU usage with Introne via timing pings (he has CPU clustering + GPU work queued)
- (Shahaan) 📋 RQ2 culturally-grounded prompt axes (warmth/authority/guilt/transactional) DEFERRED to second wave; base 4-prompt runs go first
- (Shahaan) ✅ DGX Spark research doc resent to Prof. Introne (he had not seen it)
- (Shahaan) 📋 HF cache bind pattern shared by Prof. via chat: `--bind $HF_CACHE:$HF_CACHE`
- (Shahaan) 📋 Compute nodes can pull from internet but not push outbound; pre-caching still preferred for speed

### 2026-05-11
- (Shahaan) ✅ ITS expanded OrangeGrid GPU visibility for szkhan account (response to 5/7 query)
- (Shahaan) ✅ A100 80GB PCIe confirmed available: 9 nodes / 18 GPUs / driver 12.8 / CUDA cap 8.0 / 81154 MB VRAM each, fully vLLM 0.16 compatible
- (Shahaan) 🔧 Pool re-characterized: 27 usable nodes / 68 usable GPUs (A100 + A40 + L40S). RTX 6000 / RTX 5000 / GTX 1080 Ti excluded (old drivers / too small)
- (Shahaan) 🔧 OrangeGrid strategy LOCKED to Option A: single-GPU A100 80GB per 70B/72B model (TP=1, no tensor parallelism). Matches Prof's 4/23 plan
- (Shahaan) 🔧 TP=2 across L40S/A40 (proposed 5/7) shelved as fallback contingency if A100 contention becomes blocker
- (Shahaan) ✅ 5/7 hello-world test confirmed completed (cluster 396609.0, 5/7 09:05, 1s runtime)
- (Shahaan) ⚠️ 3x CRUSH-NODE A40 machines from 5/7 recon no longer visible — cause unknown, ITS follow-up not urgent
- (Shahaan) ⚠️ Cluster topology table body in WORKPLAN still shows 5/7 stale entries; cleanup pass deferred to next session

### 2026-05-07
- (Shahaan) ✅ First SSH into OrangeGrid via SU RDS (login: its-og-login5.syr.edu, Condor 9.8.0, Ubuntu 20.04)
- (Shahaan) ✅ Empirical recon (M8.T3) complete: GPU pool fully characterized, ClassAd schema discovered (Condor 9.8 uses CUDA-prefix attrs), Singularity 3.7.1 + Apptainer 1.1.3 confirmed, no Lmod, PREEMPT=False, JOB_DEFAULT_REQUESTDISK=DiskUsage (auto)
- (Shahaan) ✅ Submission pipeline validated end-to-end (cluster 396606 submitted cleanly, condor_q -better-analyze works, partitionable-slot contention understood)
- (Shahaan) ❌ NO A100s in visible OrangeGrid pool (3 separate filters returned empty). 4/30 entry mentioning A100 80GB was inherited assumption, not verified
- (Shahaan) 🔧 Cluster topology table requires correction: actual fleet is 3x A40 (4 GPUs each, 45GB, drv 12.0) + 11x L40S (2 GPUs each, 45GB, drv 12.8) + 29x RTX 6000 (22GB, drv 11.6, vLLM-incompatible). Effective pool 14 nodes / 34 GPUs
- (Shahaan) 🔧 OrangeGrid strategy revised from 4/23: TP=2 across two L40S or A40 GPUs same node for 70B/72B (90GB total VRAM, matches Apophis 96GB envelope); TP=1 single GPU for 32B/24B
- (Prof. Introne) ⚠️ Asserts A100s exist on OrangeGrid despite our recon. ITS query sent to researchcomputing@syr.edu (cc'd Introne) asking about partition/pool visibility. TP=2 strategy conditional on ITS response
- (Shahaan) 📄 Meeting prep doc generated (Meeting_Prep_2026-05-07.docx) — one-page bulleted brief

### 2026-04-30
- (Shahaan) 📋 Weekly meeting with Prof. Introne — 8:30am
- (Shahaan) 🆕 OrangeGrid access approved 4/29 by Dan/ITS (Syracuse Research Computing). Login: its-og-login5.syr.edu via SU RDS.
- (Shahaan) 🔧 HPC strategy CLARIFIED: TWO clusters, not one. Apophis = Prof. Introne's lab box (no Condor, dev/test). OrangeGrid = ITS HTC cluster (HTCondor, hundreds of GPUs via SUrge, production target). Prof's 4/23 batch-script advice was for OrangeGrid all along.
- (Shahaan) 🔧 Apophis batch config REVERTED to TP=2 / request_gpus=2 / vram=40000 for 70Bs (4/30 morning edits applied OrangeGrid-style single-GPU config to Apophis, broke working setup, reverted same day)
- (Shahaan) 🆕 M8 — OrangeGrid Setup & Production Runs milestone created (CURRENT PRIORITY along with M4.T11 Apophis test validation)
- (Shahaan) 🆕 M9 — RQ3 Viability Scoping milestone created (4-hour timebox: OLMo/Tülu, Nemotron, Llama-family, Qwen variants, Zephyr)
- (Shahaan) 🆕 M6.T9 — Locate LLM-as-jury GLOBE framework preprint
- (Shahaan) 🆕 M6.T10 — Targeted Jinfen-aligned architecture literature search (RQ4 backup, do not cite dissertation)
- (Shahaan) 🆕 M6.T11 — Find formal citations for new prompt axes (warmth/authority/guilt/transactional)
- (Shahaan) 🆕 M1.T23 — Fix yaml.dump() API key mangling bug from run_015
- (Shahaan) ✅ M2.T18 — YAML apostrophe escaping in config3.yaml (group's, object's)
- (Shahaan) ✅ M1.T21 — fcat CLI tool deployed to ~/myenv/bin/fcat
- (Shahaan) ✅ M1.T22 — sanity_check.py built
- (Shahaan) ✅ M6.T7 — Prompt valence literature deep-dive (4 papers added)
- (Shahaan) ✅ M6.T8 — Training corpora composition research for 3 validated models
- (Shahaan) ✅ M4.T10 — Apophis 70B AWQ pipeline VALIDATED for all 3 models
- (Shahaan) 🔧 M2.T14, M3.T12, M3.T13 marked DONE (completed in milestone doc submitted Mar 12, retroactively logged)
- (Shahaan) 🔧 M4.T8 — merge_results.py marked done-but-removed (merging done locally, not on HPC)
- (Shahaan) ⏸️ M5 — explicitly flagged BLOCKED ON BUDGET APPROVAL since Mar 18 (~6 weeks)
- (Shahaan) ⏳ M7.T7 — submission deadline marked OPEN-ENDED (April target slipped)
- (Shahaan) ❌ M7.T2 (one-pager) — marked NOT NEEDED (RQ structure now serves this purpose)
- (Shahaan) 🆕 DGX Spark deliverable IN PROGRESS — local deployment comparison notes for businessman side conversation
- (Shahaan) 🔧 Cost summary table flagged PENDING UPDATE (current prices stale)
- (Shahaan) 🔧 7-model roster table updated to per-cluster validated/planned tiers
- (Shahaan) ❌ Safety / fine-tuning policy NOT added as analytical axis (Prof. Introne floated and rejected at end of 4/30 meeting)

### 2026-04-23
- (Shahaan) 📋 Weekly meeting with Prof. Introne — RQ restructure
- (Shahaan) 🔄 RQ structure consolidated from 5 → 3 RQs (per transcript), then expanded back to 4 in revised draft (architectural retained as RQ4):
  - RQ1: Cultural corpus effect on annotation (descriptive + main thesis)
  - RQ2: Prompt framing × cultural background (Prof's favorite)
  - RQ3: Corpus vs post-training fine-tuning (CONDITIONAL)
  - RQ4: Architectural influence (RQ3 backup framing)
- (Shahaan) 📄 CAMEL_RQ_v2.docx produced (3-RQ version with hypotheses, replaces/subsumes notes, dropped section)
- (Shahaan) 📄 CAMEL_Literature_to_RQ_Mapping.docx produced (25 papers mapped to RQ buckets, 3 tables, gaps section)
- (Shahaan) 🔧 Annotability hierarchy DROPPED as standalone RQ
- (Shahaan) 🔧 Zero-shot vs few-shot ABSORBED into RQ2 as sub-hypothesis
- (Shahaan) 🔧 Paper framing reframed as practitioner decision resource (not ground-truth claim)
- (Shahaan) 📋 Prof advised against over-indexing on Rathje et al. 2024 as primary framing paper
- (Shahaan) 📋 Paper submission timeline signal shifted to "early summer" (was mid-late April)

### 2026-04-03
- (Shahaan) 📋 Weekly meeting with Prof. Introne (Friday 8:30am)
- (Shahaan) 📋 Pre-meeting prep included: HPC pipeline cheat sheet, Apophis vs HPC comparison table, literature-to-pipeline mapping doc

### 2026-04-02
- (Shahaan) ✅ M1.T21 — Built fcat CLI tool from scratch (Prof. Introne's suggestion); deployed to ~/myenv/bin/fcat
- (Shahaan) ✅ M1.T22 — sanity_check.py built for pipeline readiness verification
- (Shahaan) 🔧 config3.yaml created from hpc/config3.yaml; column names corrected ("Text Number"/"Text" → "Number"/"text") via sed
- (Shahaan) 🔧 hpc/config2.yaml restored from base
- (Shahaan) ✅ M2.T18 — Two YAML parsing errors fixed (unescaped apostrophes in single-quoted strings: group's, object's)
- (Shahaan) 🐛 run_015_config3_test issues: only 9 texts loaded (wrong sample_file path); 401 Unauthorized on models after Llama (yaml.dump() in update_config_model() mangling API key on file rewrite — added as M1.T23, NOT YET FIXED)
- (Shahaan) ✅ M6.T7 — Prompt valence/incentive literature deep-dive: Wang 2025 (Neurocomputing), Felix-Pena 2025 (NeurIPS Workshop), Yin 2024 (SICon@EMNLP), Bulté & Rigouts Terryn 2025 (CL/MIT Press)
- (Shahaan) ✅ M6.T8 — Training corpora composition research: Llama 3.3 (~95% English/15T/US-RLHF), Qwen 2.5 (18T/Chinese+English/Alibaba), DeepSeek-R1-Distill (14.8T base + 800K distillation/dual Chinese)
- (Shahaan) 📄 4 Word docs produced: literature-to-pipeline mapping (full + 2-page), HPC pipeline cheat sheet, Apophis vs HPC comparison table

### 2026-03-26
- (Shahaan) 📋 Weekly meeting with Prof. Introne — lit review presentation
- (Shahaan) ✅ M6.T3 — Lit review presented to Prof. Introne (25 papers across 2 spreadsheets)
- (Shahaan) 🆕 Prompt strategy axis EXPANDED: new culturally-grounded axes proposed (positive incentive/warmth, power/authority, guilt/emotional manipulation, businesslike/transactional) — not just negative incentive. Power dynamics differ East vs West.
- (Shahaan) 💡 KEY INSIGHT: "The Matrix" — instrumental prompt axes × 25 constructs × model training culture = novel contribution space. Ontological vs processual differences distinguished.
- (Shahaan) 🆕 Prof offered Shahaan his own HPC account
- (Shahaan) 🆕 HPC batch strategy: disk space constraint may require running model subsets sequentially, clearing between batches
- (Shahaan) 🆕 Research TODO: investigate training corpora cultural fingerprinting per model (became M6.T8)

### 2026-03-25
- (Shahaan) ✅ M1.T20 — Claude Code native v2.1.79 installed, authenticated (personal Pro/Max), CLAUDE.md deployed with guardrails
- (Shahaan) ✅ M1.T15 — /DATA migration executed: samples/, outputs/, hpc/chunks/ moved to /DATA/szkhan/camel/, symlinks created
- (Shahaan) ✅ M1.T16 — GitHub cleanup pushed: .gitignore updated, README.md added, outputs removed from tracking, test_data/ with small samples preserved
- (Shahaan) ✅ M1.T17 — hpc/merge_results.py removed from repo (merging done locally)
- (Shahaan) 📋 HPC_CAMEL_README.md updated: Singularity v0.16.0, real corpus paths, symlink structure, GPU routing fixes, removed merge_results refs
- (Shahaan) 📋 CLAUDE.md corrected: fixed acronym, added guardrails, accurate env info, data locations, model roster
- (Shahaan) 🔧 .bashrc restored from /etc/skel/ after markdown paste corruption, force_color_prompt re-enabled

### 2026-03-24
- (Shahaan) ✅ M6.T1 — Systematic literature search completed (arxiv, Google Scholar, Semantic Scholar, ACL Anthology, PNAS, Nature)
- (Shahaan) ✅ M6.T2 — Annotated bibliography produced: 2 Excel spreadsheets, 25 papers total, 17 columns per paper
- (Shahaan) ✅ M6.T4 — Rathje et al. 2024 read in full detail (Purity F1=0.144, framed as "old news" baseline)
- (Shahaan) ✅ M6.T6 — Five cross-cutting themes identified from literature
- (Shahaan) ✅ M7.T1b — Five research questions drafted from lit review findings
- (Shahaan) 📋 CAMEL_Literature_Review.xlsx: 15 papers (5 prof-referenced + 10 new) with full metadata and CAMEL connections
- (Shahaan) 📋 CAMEL_Lit_Review_Tier1.xlsx: 10 Tier 1 papers (Nature, ICML, NeurIPS×2, ICLR, PNAS, EMNLP×3, TACL)
- (Shahaan) 📋 Prof-referenced papers expanded: added CAMEL_V1 (Zewail/Atari 2025) and Atari "Which Humans?" (2023) to lit review

### 2026-03-19
- (Shahaan) 📋 Weekly meeting with Prof. Introne — lit review pivot, HPC 2-week hold
- (Shahaan) 🆕 M6 — Literature Review & Subject Matter Expertise milestone created (CURRENT PRIORITY)
- (Shahaan) 📋 M6.T3 deliverable: present lit review draft at March 26 meeting
- (Shahaan) ⏸️ M4 — HPC batch processing on hold (cluster full, ~2 weeks)
- (Shahaan) ⏳ M7.T2 — One-pager on thesis deferred until lit review is substantive (was M6.T3)
- (Shahaan) 🆕 M1.T15 — /DATA storage migration planned (symlink samples/outputs to /DATA/szkhan/camel/)
- (Shahaan) 🆕 M1.T16 — GitHub cleanup: .gitignore, READMEs, API key audit
- (Shahaan) 🆕 M1.T17 — Remove merge_results.py from hpc/ (merging local only)
- (Shahaan) 🆕 M1.T18 — Prune HPC chunks to 100-text test chunk only
- (Shahaan) ✅ M1.T19 — Claude Code approved for Apophis server
- (Shahaan) 📋 Model scope confirmed: Llama, Qwen, DeepSeek (three for now)
- (Shahaan) 📋 Paper legwork responsibility acknowledged — Shahaan presents findings, not profs

### 2026-03-18
- (Shahaan) 📧 Prof. Atari responded to milestone doc — paper thesis direction established
- (Shahaan) 🆕 M6.T3 — One-pager on paper thesis added as highest priority task (later DEFERRED, now NOT NEEDED)
- (Shahaan) 📋 M5 reordered: cheaper API models first per Atari directive
- (Shahaan) 🆕 M5.T8 — LLM vs simpler NLP baseline comparison added per Atari
- (Shahaan) 🆕 M6.T9 — Read Kristensen-McLachlan et al. 2025 for positioning
- (Shahaan) ✅ M6.T2 — Milestone doc confirmed received by Prof. Atari
- (Shahaan) ✅ M2.T16 — System/user prompt split for vLLM prefix caching (tested: +7–9% RPM)
- (Shahaan) ✅ M2.T17 — --no-split flag for legacy mode
- (Shahaan) ✅ M2.T15 — Pipeline output switched to Feather format, pyarrow installed
- (Shahaan) ✅ M1.T13 — Prompt caching implemented and benchmarked
- (Shahaan) ✅ M4.T1 — HTCondor batch submit scripts built (per-model with GPU routing)
- (Shahaan) ✅ M4.T2 — GPU routing by model size (70B→2 GPU, 32B/24B→1 GPU)
- (Shahaan) ✅ M4.T4 — Prefix caching tested on Apophis (+7% Llama, +9% Qwen)
- (Shahaan) ✅ M4.T5 — 56K corpus converted to Feather, split into 12 × 5,000 chunks
- (Shahaan) ✅ M4.T6 — camel_annotate_hpc.py built (self-contained HPC annotation)
- (Shahaan) ✅ M4.T7 — generate_batch_jobs.py built (per-model submit generation)
- (Shahaan) ✅ M4.T8 — merge_results.py built (later removed; merging done locally)
- (Shahaan) ✅ M4.T9 — Test chunk (100 texts) created for HPC validation
- (Shahaan) 📋 Weekly report prepared for Prof. Introne meeting

### 2026-03-12
- (Shahaan) ✅ Llama 3.3 70B upgrade completed (awq_marlin confirmed)
- (Shahaan) ✅ evaluate.py "Keep the 0" fix applied
- (Shahaan) ✅ Ollama vs vLLM benchmark completed (4.4x–5.6x speedup)
- (Shahaan) ✅ M6.T2 — Milestone doc submitted to Prof. Introne, forwarded to Atari
- (Shahaan) ✅ M2.T14, M3.T12, M3.T13 — 25×9 F1 table, prompt templates included in milestone doc

### 2026-03-05
- (Shahaan) ✅ M3.T8–T9 — 50-sample benchmark completed (run_005_testing_50_r2)
- (Shahaan) ✅ M3.T10–T11 — East-West divergence + DeepSeek over-interpretation documented
- (Shahaan) ✅ M5.T1 — Full API cost analysis completed
- (Shahaan) 📋 Meeting with Prof. Introne — key action items assigned

### 2026-03-04
- (Shahaan) ✅ M2.T11 — Fixed input token overflow (per-model max_tokens)
- (Shahaan) ✅ M2.T12 — Run directory organization (outputs/run_NNN_name/)
- (Shahaan) ✅ M1.T11 — Checkpoint/resume implemented via skip_existing + glob scan
- (Shahaan) 🆕 LLM Landscape research: frontier API pricing, Arabic LLMs, 2026 leaderboards

### 2026-03-03
- (Shahaan) ✅ M3.T1–T4 — Gold standard loaded, evaluate.py built
- (Shahaan) ✅ M3.T6 — First 9-text eval results
- (Shahaan) ✅ M2.T10 — Fixed parse_yes_no for DeepSeek <think> tags
- (Shahaan) 🔍 M3.T10 — Qwen ultra-conservative behavior confirmed as legitimate finding

### 2026-03-02
- (Shahaan) ✅ M1.T6 — nvcc found at cuda-12.2, CUDA_HOME fixed
- (Shahaan) ✅ M1.T7–T9 — Subprocess env, model sync, Ollama disabled, max_tokens
- (Shahaan) ✅ M3.T5 — First successful 3-model automated run

### 2026-03-01
- (Shahaan) ✅ M1.T3 — Python 3.8 → 3.11 upgrade, full stack installed
- (Shahaan) ✅ M1.T4 — Jupyter Lab persistent workspace configured
- (Shahaan) ✅ M1.T5 — vLLM 0.16.0 compat fixes (flags, paths, GPU util)

### 2026-02-17
- (Shahaan) ⏳ M2.T1 — Early F1 results: Llama=0.339, Qwen=0.192, Sonnet 4=0.222

### 2026-02-16
- (Shahaan) ⏳ M2.T1 — Initial pilot with camel_annotation_demo.py, 6 prompting strategies
