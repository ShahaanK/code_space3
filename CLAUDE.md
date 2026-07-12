# CLAUDE.md — CAMEL Project Guardrails for Apophis

This file gives Claude Code (and any LLM tool) the context, conventions, and
constraints for working in `~/code_space3` on the Apophis server. Read this
before making any changes to the project.

---

## Project Snapshot

**CAMEL** = Cultural and Moral Expressions in Language. The pipeline benchmarks
LLMs as annotators of 25 psychological and cultural constructs (moral
foundations, cultural dimensions, behavioral markers) against a 56,796-text
human-annotated corpus. The north-star comparison is **East vs West**.

- **Author / first author:** Shahaan Khan
- **Advisor:** Prof. Joshua Introne (Syracuse University iSchool)
- **Collaborator:** Prof. Mohammad Atari (UMass Amherst, Culture and Morality Lab)
- **Degree:** MS Applied Human-Centered AI, Syracuse iSchool
- **Submission target:** Open-ended (April 2026 target slipped, new date TBD with Introne)

---

## Project Guardrails

### Never (without explicit confirmation)

- Modify or delete files in `~/myenv/` (Python virtual environment)
- Reinstall vLLM, torch, or transformers — current versions are pinned to a working configuration
- Stop, restart, or reconfigure the vLLM server process
- Modify `config2.yaml` or `config3.yaml` (master prompts/labels). Note: `config3.yaml` is the OrangeGrid production config and the DeepSeek block's `temperature`/`max_tokens` in it are NOT read on the HPC path (see HPC section) — decoding params come from code, not this yaml.
- Modify files in `hpc/` (production batch scripts — now the active work area; changes must still be shown as a diff and confirmed before applying)
- Re-add `--kv-cache-dtype fp8_e4m3` to the vLLM invocation in `hpc/camel_annotate_hpc.py`. It is deliberately commented out to keep the V1 engine active (see V0/V1 note). Re-adding it silently forces the slow V0 engine.
- Change the DeepSeek `MODEL_OVERRIDES` entry in `hpc/camel_annotate_hpc.py` (`max_tokens: 2048`, `repetition_penalty: 1.15`) without confirmation — it is under active tuning (see Active Problems).
- Apply OrangeGrid HPC config (TP=1, single GPU, A100 targeting) to Apophis, or vice versa. Doing so on 4/30 broke the validated TP=2 setup and was reverted same day.
- Delete files in `samples/` or `outputs/` (symlinked to `/DATA`)
- Push to git without confirmation. Apophis CAN push (SSH remote, `git@github.com:ShahaanK/code_space3.git`); it is the commit path for the project. OG's remote is HTTPS with a read-only PAT and cannot push (see Git & Sync).
- Run `pip install` or any package installation
- Execute destructive commands (`rm`, `mv`, force-push, mass overwrite) on data files
- Cite Prof. Jinfen Li's dissertation as a reference. (Use her published refs if she shares them; the dissertation itself is off-limits.)
- Commit large data files (`.feather`, full corpus CSVs) — `.gitignore` excludes `.feather`; result CSVs should be treated the same (keep them in `/DATA` or ignored, not tracked)
- Hardcode the API key in any committed file
- Paste markdown directly into the terminal (corrupted `~/.bashrc` once on 3/25; restored from `/etc/skel/`). Use `nano`/`vim` for in-terminal edits.

### Always

- Read this CLAUDE.md and the latest WORKLOG entry before making changes
- Show a diff or summary of changes before applying edits to pipeline files
- Update admin/WORKLOG.md and admin/WORKPLAN.md at the end of each working session
- Validate Python after edits: `python -c "import ast; ast.parse(open('FILE').read())"`
- Validate YAML after edits: `python -c "import yaml; yaml.safe_load(open('FILE'))"`
- Use grep + sed for targeted fixes; preserve old versions as `*_old.py` in git
- Confirm which cluster a file targets before editing it (Apophis and OG diverge; see Git & Sync)
- When in doubt, ask before acting

---

## Compute Environment — TWO CLUSTERS

**Important:** This project uses two distinct clusters with different conventions.
Never apply config from one to the other without checking.

### Apophis (this machine — Prof. Introne's lab box)

- Host: `ist-rs-introne` / `apophis.ischool.syr.edu`
- Hardware: 48 cores, 251 GB RAM, **2× RTX A6000 (48 GB each, sm_86 Ampere)**
- Total VRAM: 96 GB (when using both GPUs)
- Scheduler: **None.** Direct vLLM via `run_models.py` or `wrapper_<model>.sh`.
- NOT a Condor cluster.
- Storage: `/home/szkhan` (NFS) plus `/DATA/szkhan/camel/` (local NVMe, 7.3 TB)
- vLLM: **0.16.0**, V1 engine (default), API key `sk-local-shahaan` on port 8900 (local only, not a real secret)
- Access: SSH via Syracuse RDS, port-forwarded Jupyter Lab on 8891 (started with nohup from `~/code_space3`)
- Role: **Dev/test only.** 50-sample benchmarks, prompt experiments, validation.

### OrangeGrid (Syracuse ITS HPC — production target, OPERATIONAL as of May 2026)

- Login: `its-og-login5.syr.edu` via SU RDS
- Scheduler: **HTCondor 9.8.0** with `+request_gpus = N` ClassAd syntax
- Effective usable pool: **27 nodes / 68 GPUs** — 9× A100 80GB PCIe (CUDA 12.8), 7× A40, 11× L40S. RTX 6000/5000 nodes are vLLM-incompatible (CUDA 11.6) and excluded.
- vLLM: **0.10.1.dev1+gbcc0a3cbe** (commit bcc0a3c, from Introne-provided SIF `vllm-openai.sif`). Lower than Apophis because CUDA 12.8 caps vLLM at 0.10.x.
- Container: Singularity 3.7.1 / Apptainer 1.1.3. Use `python3` (not `python`) inside the SIF.
- Storage: NetApp `/home/szkhan` (4 TB), auto-mounted on compute nodes
- Login nodes are SUBMIT-ONLY: no Jupyter, no IDE, no long tmux
- Status: **Stood up and validated (May 2026).** 100-text Llama test (job 469622) and a 100-text multi-model characterization (Llama + Qwen) have run successfully. Full 56K production wave NOT yet launched (gated — see Active Problems).
- Role: **Production 56K runs** across the validated models

### vLLM V0 vs V1 Engine (OrangeGrid)

The single biggest OG throughput lever. vLLM 0.10.1.dev auto-falls-back to the
legacy V0 engine when `--kv-cache-dtype fp8_e4m3` is set (V1 does not support that
flag in this build). Removing the flag unlocks the V1 engine and tripled Llama
throughput from **63 RPM (V0) to 224 RPM (V1)** on the same A100, exceeding the
Apophis reference. The flag is therefore commented out in `camel_annotate_hpc.py`
and must stay that way. Removing fp8 is a pure win, not a precision trade-off.
V1 was confirmed annotation-neutral (identical macro F1 across all three prompts).

### HPC Configuration — TWO DISTINCT CONFIGS

When editing batch scripts, the right config depends on the cluster:

| Setting | Apophis (validated) | OrangeGrid (validated in production) |
|---|---|---|
| Tensor parallelism | TP=2 (both A6000s) | TP=1 (single A100 80GB) |
| `request_gpus` | 2 (for 70Bs) | 1 |
| `+request_gpus` | n/a (no Condor) | 1 (ClassAd syntax; slot START checks the underscored form) |
| GPU targeting | n/a | `Requirements = (CUDADeviceName == "NVIDIA A100 80GB PCIe") && (TARGET.GPUs >= 2)` |
| Argus opt-in | n/a | `+wantsArgusNode = True` |
| File transfer | n/a | `should_transfer_files = NO` (do NOT add `when_to_transfer_output = ON_EXIT` — contradicts it, errors at dry-run) |
| HOME | inherited | must be explicit in `environment` ClassAd |
| Model selection | n/a | `CAMEL_MODEL=<full HF path>` in `environment` (the template `wrapper.sh` defaults to Llama otherwise); generated per-model wrappers bake `--model` directly |

`TARGET.GPUs >= 2` is a proxy for "both A100s on the node are free," which avoids
the GPU-memory contention that stalls jobs when HTCondor's slot bookkeeping
diverges from `nvidia-smi` reality.

---

## Project Layout

```
~/code_space3/
├── adapter.py              # OpenAI-compatible client (any provider; Apophis path)
├── prompt_builder.py       # Conditional prompt assembly + system/user split
├── run_annotation.py       # Main runner (sync/async/threaded, Feather output)
├── run_models.py           # Multi-model vLLM orchestrator
├── evaluate.py             # F1-macro/micro per label/model/prompt
├── benchmark_workers.py    # Worker count throughput sweep
├── update_config_models.py # Config sync utility (HAS A KNOWN BUG — see below)
├── config2.yaml            # Primary config — verbatim CAMEL definitions
├── config3.yaml            # Variant config (annotation-guide changes; OG production config)
├── samples/  →  /DATA/szkhan/camel/samples/   (symlink)
├── outputs/  →  /DATA/szkhan/camel/outputs/   (symlink)
├── test_data/              # Small sample files for cloning (in git)
├── hpc/
│   ├── camel_annotate_hpc.py     # Self-contained annotation (MODEL_OVERRIDES holds DeepSeek max_tokens + repetition_penalty; decoding params are here, NOT in config)
│   ├── generate_batch_jobs.py    # Per-model HTCondor submit generation (bakes full HF path into wrapper)
│   ├── split_data.py             # Corpus → Feather chunks (--chunk-size CLI arg)
│   ├── wrapper.sh                # Template wrapper — reads CAMEL_CONFIG override, converts CUDA_VISIBLE_DEVICES UUIDs → integer indices, sets HF offline env
│   ├── wrapper_llama-3.3-70b.sh  # Generated per-model wrapper (bakes --model)
│   ├── test_job.sub              # Apophis-style test submission (TP=2)
│   ├── test_job_og.sub           # OrangeGrid single-model test (A100, TP=1, config3)
│   ├── test_job_multi.sub        # OrangeGrid multi-model queue-from-table submit
│   ├── condor_alert.sh           # Slack watcher (fires on job 005/012/009 events)
│   ├── config2.yaml              # Columns: Number, text (56K corpus)
│   ├── config3.yaml              # OG production config (Number/text; apostrophes escaped)
│   ├── chunks/  →  /DATA/szkhan/camel/hpc_data/chunks/   (symlink on Apophis; real dir on OG)
│   └── HPC_CAMEL_README.md
├── admin/
│   ├── WORKPLAN.md             # Milestones + reference tables + changelog
│   ├── WORKLOG.md              # Reverse-chronological narrative
│   └── OrangeGrid_Quickstart.md  # OrangeGrid connection, job submission, and cluster reference
├── README.md               # Project overview
└── CLAUDE.md               # This file (local to Apophis; not on OG)
```

`merge_results.py` was removed 3/25 — merging is done locally on Apophis,
not on HPC nodes.

`samples/` and `outputs/` are symlinks to `/DATA` because `/home` was at 67%
capacity. Don't move data back to `/home`. On OG the `chunks/logs/results`
symlinks are replaced with real directories (they arrive broken from the clone).

---

## Git & Sync Topology

This bit the project repeatedly and is worth internalizing.

- **Apophis remote:** SSH (`git@github.com:ShahaanK/code_space3.git`) — can push. This is the commit path.
- **OrangeGrid remote:** HTTPS with a Contents: Read-only PAT — cannot push. Regenerating it with Read+Write is an open task (M8.T17).
- Because OG could not push, `hpc/` code edits made during production setup (config3 apostrophe fix, wrapper CAMEL_CONFIG + UUID handling, submit files, the DeepSeek override) lived only on OG and drifted from Apophis and GitHub. As of 2026-07-06 those were promoted OG→Apophis and committed to GitHub from Apophis.
- **Rule:** OG is authoritative for `hpc/` code when in doubt; verify with `diff` before promoting either direction. Never blind-rsync a whole file across boxes without diffing first, or you can revert the V1 unlock or the apostrophe fix.
- CLAUDE.md is local to Apophis (not tracked/synced). It only needs to live where Claude Code runs.

---

## Environment

- **Active venv:** `~/myenv` (Python 3.11.13)
- Has everything: vLLM 0.16.0, torch, openai, pandas, pyarrow, Jupyter
- Activate: `source ~/myenv/bin/activate`
- `~/.venv-vllm` is redundant; ignore it
- `~/myenv38_backup` is the old Python 3.8 env; do not use

`fcat` (Feather inspector) is at `~/myenv/bin/fcat` — has subcommands
`head/tail/info/describe/columns/shape/cols/sample/unique/counts/query/grep/labels/export`.

---

## Key Commands

### Run annotation pipeline

```bash
# Activate environment first
source ~/myenv/bin/activate

# Default mode (system/user split for vLLM prefix caching)
python run_annotation.py --config config2.yaml --workers 16

# Legacy single-message mode (avoid in new runs)
python run_annotation.py --config config2.yaml --workers 16 --no-split

# Dry run to preview prompts without API calls
python run_annotation.py --config config2.yaml --dry-run
```

### Automated multi-model runs on Apophis

```bash
# Run all enabled models sequentially (starts vLLM, runs pipeline, cycles to next)
python run_models.py --config config2.yaml --workers 16

# Run specific models by index
python run_models.py --models 0,2 --workers 16
```

### Evaluate results against gold standard

```bash
python evaluate.py --results outputs/camel_results_*.feather --gold samples/50_random_samples_ans.csv
python evaluate.py --verbose  # Per-label breakdown
```

### HPC batch processing (HTCondor — OrangeGrid)

```bash
cd hpc/
python split_data.py --input <corpus.feather> --chunk-size 1000   # → chunks/ + chunk_manifest.json
python generate_batch_jobs.py --cluster orangegrid --model deepseek-r1-32b   # per-model submit + wrapper
condor_submit test_job_og.sub                          # single-model test
condor_q                                               # monitor
condor_q -better-analyze <jobID>                       # debug if jobs sit idle
condor_ssh_to_job <jobID>                              # attach to a running job (nvidia-smi, curl the server)
```

`condor_history` ages out in ~1 week; rely on `logs/*.out` and `logs/*.log`
for older job forensics. For held jobs, prefer hold-edit-release over
rm-resubmit (Introne's unverified intuition: rm-resubmit may deprioritize the user).

### Test job validation

`generate_batch_jobs.py` requires `chunks/chunk_manifest.json`. If using only
`test_chunk.feather` (no full split), hand-write the manifest with required
fields: `total_chunks`, `chunk_size`, `total_unprocessed`, `chunks` array
where each entry has `chunk_id`, `filename`, `rows`, `original_start_index`,
`original_end_index`.

---

## Architecture

### Core Pipeline Components

- **run_annotation.py**: Main runner — iterates over models × prompts × texts × labels.
  Supports sync, threaded (joblib), and async execution. Outputs Feather format.
  Checkpoint/resume via `skip_existing` scan.

- **adapter.py**: OpenAI-compatible client for any provider (vLLM, Ollama, OpenRouter).
  Implements `call_model()` and `call_model_async()` with retry logic and
  system/user message splitting for prefix caching. **Apophis path only** — the HPC
  runner does not use adapter.py.

- **prompt_builder.py**: Assembles prompts from config templates.
  `build_prompt_split()` returns (system_prompt, user_prompt) tuple for vLLM prefix
  caching; `build_prompt()` returns single string for legacy mode.

- **evaluate.py**: Computes precision/recall/F1 per label, macro/micro averages.
  Handles "strict" (majority agreement) and "lenient" (any annotator) gold standard.
  Zero-support labels ARE included in macro F1 (prof directive: "Keep the 0").
  Runs on Apophis, where the gold key lives.

- **run_models.py**: Orchestrates multi-model runs — starts vLLM with correct
  quantization/TP settings, waits for readiness, runs pipeline, kills server,
  cycles to next model.

- **camel_annotate_hpc.py**: Self-contained HPC runner. Starts vLLM via Singularity,
  processes one chunk, saves Feather. Builds its request payload by hand over urllib
  (no OpenAI SDK), so vLLM-specific params like `repetition_penalty` go straight into
  the payload dict as top-level keys (no `extra_body` needed). Sampling params come
  from `MODEL_OVERRIDES` constants and the `call_vllm` signature, NOT from the config
  YAML. Output path is a positional CLI argument, not read from config `output_dir`.

- **benchmark_workers.py**: Worker count throughput sweep for optimizing parallelism.

- **update_config_models.py**: Syncs config model entries with LOCAL_MODELS dict.
  ⚠️ Has a known bug (see Known Bugs below).

### Configuration

`config2.yaml` is the master config (DO NOT EDIT WITHOUT CONFIRMATION). It defines:

- `runtime`: Sample file path, output settings, parallelism options
- `providers`: Base URLs and API key env vars for each backend
- `models`: Model names, providers, temperatures, max_tokens, enabled flags
- `prompts`: Templates with placeholders for `{text}`, `{label_name}`, `{label_definition}`, etc.
- `labels`: 25 labels with definitions, markers, and examples (verbatim from CAMEL Annotation Guide)
- `annotation_guidelines`: Shared instructions injected into all prompts

`config3.yaml` is the annotation-guide-aligned variant and is the config the OG
production jobs read (`CAMEL_CONFIG=config3.yaml`). Its `models` block controls
which prompts/labels loop via `enabled`, but on the HPC path its per-model
`temperature`/`max_tokens` are ignored — those come from `camel_annotate_hpc.py`.

### System/User Prompt Split

The pipeline splits prompts at `=== TEXT ===` marker:

- **System message**: Role, guidelines, label definition (static per prompt × label, cached by vLLM)
- **User message**: Just the text to annotate (varies per text)

This enables vLLM's automatic prefix caching (+7% Llama, +9% Qwen throughput at
9-text scale; ~0% on DeepSeek-R1, whose bottleneck is output generation not input).

### Output Format

Results are Feather files (`.feather`) in wide format:

- Metadata: `text_id`, `prompt_id`, `prompt_name`, `model`, `provider`, `temperature`
- 25 label columns: 1 = YES, 0 = NO, -1 = UNCLEAR/ERROR
- 25 `response__<label>` columns with raw model responses

Feather has no append: results accumulate in memory, full-file overwrite every 50 rows.

---

## Important Patterns

### Response Parsing

`parse_yes_no()` exists in BOTH `run_annotation.py` (Apophis path) and
`camel_annotate_hpc.py` (HPC path, ~line 163). Behavior:

- Strips `<think>...</think>` blocks from DeepSeek-R1 reasoning chains
- Checks both head and tail of response for YES/NO patterns
- Returns -1 for unclear or error responses (this is how DeepSeek degeneration
  shows up in results — as a high -1 rate, silently, not as a raised error)

### Per-model `max_tokens` and sampling (HPC — camel_annotate_hpc.py MODEL_OVERRIDES)

- Llama / Qwen: `max_tokens` 256 (YES/NO output is small), no penalty
- **DeepSeek: `max_tokens` 2048, `repetition_penalty` 1.15** (updated 7/6, was 1024/none).
  temperature stays 0 for all models (deterministic, bias-faithful annotation).
  The penalty is under active tuning — see Active Problems.

### Column Conventions

- 50-sample files (`samples/sample_prompts.xlsx`): columns are `Text Number`, `Text`
- Full 56K corpus (`samples/CAMEL_cleaned_data_complete.feather`): columns are `Number`, `text`
- `hpc/config2.yaml` and `hpc/config3.yaml` have `Number`/`text` (56K-aligned)
- Root `config2.yaml` may use `Text Number`/`Text` (50-sample-aligned)
- **Don't mix.** Mismatch silently produces wrong results.

### Resumption Support

Pipeline loads existing results and skips completed `(text_id, prompt_id, model)` tuples.
Results flush to disk every 50 rows.

---

## Model Roster

| # | Model | Bucket | Origin | Size | Status | Notes |
|---|-------|--------|--------|------|--------|-------|
| 1 | Llama 3.3 70B (`casperhansen/llama-3.3-70b-instruct-awq`) | West | Meta, USA | 70B | ✅ Validated; OG V1 224 RPM; 100-text characterized | TP=2 Apophis / TP=1 OG, max_tokens=256 |
| 2 | Qwen 2.5 72B (`Qwen/Qwen2.5-72B-Instruct-AWQ`) | East | Alibaba, China | 72B | ✅ Validated; OG 100-text characterized | max_tokens=256, `--trust-remote-code` |
| 3 | DeepSeek-R1 32B (`Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ`) | East | DeepSeek, China | 32B | ⚠️ ACTIVE PROBLEM — degenerates under deterministic decoding (see Active Problems) | max_tokens=2048, repetition_penalty=1.15, `--trust-remote-code` |
| 4 | Mistral Small 24B | West | Mistral, France | 24B | Planned | Not yet run |
| 5 | AceGPT-v2 70B | East | KAUST, Saudi Arabia | 70B | Planned | `--trust-remote-code` |
| 6 | Falcon-H1 34B | East | TII, UAE | 34B | Planned | `--trust-remote-code` |

*March RPM benchmarks (Apophis): Llama 195, Qwen 162, DeepSeek 93 — not hard rules,
throughput varies with prompt length, batch size, and concurrent load.*

The north-star comparison is **East vs West**. Specific country/org origin is
documented per model but the two buckets are the analytical axis.

Emerging finding (100-text characterization): Llama over-predicts, Qwen
under-predicts, near-identical aggregate F1 through opposite failure modes; both
collapse to near-zero F1 on the moral-foundation constructs (Care, Equality,
Loyalty, Authority, Proportionality, Purity).

Validated runs: `run_005` (Llama 3.1 50-sample, F1=0.285), `run_007` (Llama 3.3
50-sample). Quantization: INT4 AWQ via `awq_marlin` (fallback to `awq`).

---

## Research Framing — 4 RQs (revised 4/23/2026)

- **RQ1** — Cultural variability from training corpus (descriptive + main thesis)
- **RQ2** — Prompt framing × cultural background (Prof's favorite, novel) — culturally-grounded axes (warmth, power/authority, guilt, transactional) layered on 25 constructs and model training culture (Mar 26 "Matrix")
- **RQ3** — Training corpus vs post-training fine-tuning (CONDITIONAL on finding clean HF model lineages with documented variants — drop if not viable)
- **RQ4** — Architectural influence (size, quantization, MoE vs dense)

Prof. Introne formal sign-off pending. Earlier 5-RQ structure superseded.
Core thesis (per Atari): LLMs systematically fail to capture certain psychological
constructs, patterned across training culture, architecture, and prompting.
Position as a nuance extension of Rathje et al. 2024, not a refutation.

The "Matrix" (Mar 26 meeting): instrumental prompt axes × 25 constructs ×
model training culture = the novel contribution space. Ontological differences
(what models get right/wrong) vs. processual differences (how prompting changes
behavior).

---

## Active Problems (being worked)

- **DeepSeek-R1 cannot yet be cleanly annotated under a deterministic protocol.**
  This is the current top blocker on the production wave.
  - Bare greedy (temperature 0, no penalty) → repetition-loop collapse.
  - Greedy + `repetition_penalty` 1.15 → a DIFFERENT degeneration: penalty-induced
    token-level incoherence (malformed output like "whether whether", tag salad),
    yielding **36.5% unparseable (-1) on the 100-text run** even though an isolated
    single-prompt curl looked clean.
  - So two deterministic configs have now failed at the 100-text gate. The failure
    is silent (emits -1, no error), so any DeepSeek run must be checked for -1 rate.
  - Next step: try a gentler `repetition_penalty` (~1.05); if that still fails,
    escalate to the June diagnosis doc's Option 4 — either document DeepSeek as a
    reasoning-model exception in the methods (a legitimate finding about
    reasoning-distilled models), or swap the Eastern reasoning slot for a model with
    a controllable non-thinking mode (Qwen3 thinking-disabled), which changes the
    roster and cultural-lineage story and is therefore an Introne/Atari decision.
  - Do NOT launch the full corpus for DeepSeek until this resolves. Llama and Qwen
    are unaffected and can proceed.

- **F1 regression vs March, undiagnosed.** 100-text OG eval (config3, Llama 3.3):
  incentive_strict macro F1 = 0.208 vs March's 0.285 (Llama 3.1, run_005), ~27%
  relative drop. Leading hypothesis: config3's longer construct definitions.
  Alternatives: Llama 3.1→3.3 are not directly comparable baselines; sample variance
  (100 vs 50). Should be isolated (config2 vs config3 on Llama) before a full
  production wave, so a degraded config does not burn 8+ days of A100 time.

---

## Known Bugs (Open)

- **`update_config_models.py` `yaml.dump()` mangles API key on file rewrite.**
  After Llama completes a multi-model run, subsequent models get 401
  Unauthorized because the API key has been re-quoted or stripped. Diagnosed
  4/2, not yet fixed (M1.T23). Workaround: re-export `VLLM_API_KEY` between
  models or run models individually.

- **YAML apostrophe sensitivity in single-quoted strings.** Inside YAML
  single-quoted strings, an apostrophe must be doubled (`group's` → `group''s`).
  Bit `config3.yaml` on the Hate construct (`group's inferiority`, line 367) and
  the AnalyticalThinking construct (`object's behavior`, lines 491–494). Fixed on
  OG during the 5/21 marathon and committed from Apophis on 7/6. Flag if similar
  appears in new prompts.

---

## Legacy Files

- `camel_annotation_demo.py` — original monolithic script, superseded by modular pipeline
- `*_old.py` files — previous versions, kept in git history
- `~/.venv-vllm/` — redundant venv, do not use
- `~/myenv38_backup/` — Python 3.8 env, archived

---

## Current Priorities (2026-07-06)

1. Resolve the DeepSeek deterministic-annotation failure (try `repetition_penalty` 1.05; else Option 4 decision with Introne/Atari) — see Active Problems
2. Diagnose the F1 regression (config2 vs config3 on Llama; 3.1 vs 3.3 comparability)
3. Apophis re-baseline: confirm 195 RPM is API-call rate, not row rate (decides how much OG headroom remains)
4. Chunk the corpus at 1,000 texts/chunk (57 chunks) and launch the production wave for Llama + Qwen once DeepSeek is decided; submit first chunk per model, verify, then release the rest (Introne subset protocol)
5. Regenerate admin/WORKLOG.md, admin/WORKPLAN.md, and VISION.md current through today (in progress)
6. Regenerate the OG GitHub PAT with Read+Write scope (M8.T17) to end OG→GitHub drift

See admin/WORKPLAN.md for the full milestone breakdown and admin/WORKLOG.md for the
reverse-chronological narrative.

---

*Last refreshed: 2026-07-06. Replaces prior CLAUDE.md dated 2026-04-30.*
