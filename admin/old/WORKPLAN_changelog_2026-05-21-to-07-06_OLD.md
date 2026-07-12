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

