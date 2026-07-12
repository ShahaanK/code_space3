# WORKPLAN Updates: May 14 to May 19 Session Block

Apply these edits to `WORKPLAN.md` on Apophis. Sections grouped by where they belong in the existing file.

---

## Changelog entries (append to the dated changelog)

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

---

## M8 Milestone Updates

Replace the M8 task list with the following:

### Milestone 8: OrangeGrid HPC Setup & Production Runs

**Status:** Day 0 plumbing complete (5/19). Day 1 (HF model cache + repo + batch script adapt) next. Day 2 100-text test targeted before 5/21 weekly meeting.

- [x] M8.T1: Request OrangeGrid access (approved 4/29 by Dan/ITS)
- [x] M8.T2: First SSH connection to its-og-login5.syr.edu via SU RDS (5/7)
- [x] M8.T3: Empirical reconnaissance (5/7 + 5/11)
- [x] M8.T4: ITS A100 access query (resolved; A100s now visible)
- [x] M8.T5: Set up OrangeGrid environment (Miniforge 24.7.1, camel conda env Python 3.11, all deps installed)
- [x] M8.T6: vLLM SIF on OrangeGrid (Prof. Introne provided via Apophis SCP, copied 5/19; vLLM 0.10.1.dev1+gbcc0a3cbe verified)
- [ ] M8.T7: Pre-cache HuggingFace model weights to /home/szkhan (Llama 3.3 70B AWQ, Qwen 2.5 72B AWQ, DeepSeek-R1 32B AWQ; ~92GB total; run inside tmux)
- [ ] M8.T8: Clone code from GitHub to OrangeGrid; rsync chunks/ from Apophis
- [ ] M8.T8b: Claude Code SIF deep-dive: entrypoint pattern, env vars, AWQ kernel, Python path
- [ ] M8.T9: Adapt batch scripts for OrangeGrid: TP=1, request_gpus=1, A100 80GB constraint via CUDADeviceName ClassAd, HF cache bind, `python3` invocation, vLLM 0.10.1.dev API
- [ ] M8.T10: Submit OrangeGrid 100-text test (Llama 3.3 70B, A100, --no-split mode first); validate Feather output against Apophis baseline
- [ ] M8.T10b: Re-submit 100-text test with system/user prompt split; measure prefix-caching RPM delta on vLLM 0.10.1.dev (Apophis baseline was +7-9% at vLLM 0.16)
- [ ] M8.T11: Submit full 56K production runs across all 3 validated models (12 chunks × 3 models = 36 jobs); subset-based submission per Introne coordination protocol
- [ ] M8.T12: Monitor with `condor_q -better-analyze`; hot-fix held jobs in place; `condor_release` to resume
- [ ] M8.T13: Merge chunk results locally on Apophis (not OrangeGrid); evaluate at scale
- [ ] M8.T14: Statistical significance tests for East-West divergence at full corpus scale
- [ ] M8.T15: Report per-chunk runtime + GPU usage to Introne after test job (per 5/14 coordination protocol)
- [ ] M8.T16: Ongoing coordination pings with Introne during production wave (he has CPU clustering + GPU work queued)

---

## M7 Sequencing Note

Add this as M7.T1e:

- [ ] M7.T1e: RQ2 culturally-grounded prompt axes (warmth, power/authority, guilt, transactional) implementation DEFERRED to post-base-wave. First 56K production runs use existing 4 prompting strategies (zero-shot, few-shot, minimal, incentive_strict). New axes added as second wave once base baseline is locked. (Decision: 5/14 meeting, Prof. Introne agreed)

---

## Cluster Topology Reference Table: Add vLLM Version Row

Add a row to the existing cluster topology table:

| Attribute       | Apophis                           | OrangeGrid                              |
|-----------------|-----------------------------------|-----------------------------------------|
| vLLM version    | 0.16.0 (pinned, do not touch)     | 0.10.1.dev1+gbcc0a3cbe (commit bcc0a3c) |
| CUDA driver     | 12.x                              | 12.8 (caps vLLM at 0.10.x)              |

Other rows remain as in the existing table.

---

## Side Items

Mark as complete:

- [x] DGX Spark research doc, resent to Prof. Introne (he had not seen it; 5/14)

---

## Reference: Files Produced This Session

- `WORKLOG_additions.md` (dated entries 5/14 and 5/19)
- `WORKPLAN_additions.md` (this file)
- `OrangeGrid_Quickstart.md` (standalone reference doc; add to project documents)
