# WORKPLAN Updates: 2026-05-21

Append to `WORKPLAN.md` on Apophis.

---

## Changelog entries (append to dated changelog)

### 2026-05-21
- (Shahaan) ✅ M8.T7: HF model cache complete on OrangeGrid (95GB total, all 3 validated models)
- (Shahaan) ✅ M8.T8: Repo cloned to OG via Personal Access Token (rsync fallback rejected due to .venv-camel bulk)
- (Shahaan) ✅ M8.T8b: SIF deep-dive findings: vLLM 0.10.1.dev1+gbcc0a3cbe, awq_marlin kernel confirmed available at runtime, python3 (not python) required inside container, V0 engine forced by fp8 KV cache flag
- (Shahaan) ✅ M8.T9: Adapted batch script for OrangeGrid (test_job_og.sub with all 12 gotcha fixes)
- (Shahaan) 🔄 M8.T10: 100-text Llama test IN-FLIGHT (job 469622, submitted 12:44 EDT, expected completion 14:43 EDT)
- (Shahaan) 🔧 12 distinct OrangeGrid gotchas resolved in one session (see WORKLOG for full list)
- (Shahaan) 📋 Live observed rate: 65 RPM on OG A100 TP=1 vLLM 0.10.1.dev V0 (vs 195 RPM Apophis baseline, 3x slowdown)
- (Shahaan) 🔧 Arithmetic correction: each (text x prompt) = 25 API calls (one per label), not 1 request. 100-text test = 7,500 API calls.
- (Shahaan) 📋 Production_Runtime_Estimates_Rev2.docx produced with corrected math and live OG rate
- (Shahaan) 🆕 M8.T10c: V1 engine optimization experiment (remove --kv-cache-dtype fp8_e4m3 from vLLM invocation, re-measure)
- (Shahaan) 🆕 M8.T10d: Apophis re-baseline (verify 195 RPM is API calls/min not rows/min)
- (Shahaan) 🆕 M8.T10e: config2 vs config3 prompt-token impact measurement
- (Shahaan) 🆕 M8.T10f: Concurrent thread count scaling test (16 vs 32 vs 64)

---

## M8 Milestone Updates

Replace M8 task list with the following:

### Milestone 8: OrangeGrid HPC Setup & Production Runs

**Status:** First-light run executing. Pipeline end-to-end validated, but throughput is 3x below Apophis baseline. Optimization experiments needed before production wave.

- [x] M8.T1: OrangeGrid access (approved 4/29)
- [x] M8.T2: First SSH connection (5/7)
- [x] M8.T3: Empirical reconnaissance (5/7 + 5/11)
- [x] M8.T4: A100 access (resolved before 5/14)
- [x] M8.T5: Miniforge env setup (5/19)
- [x] M8.T6: vLLM SIF on OrangeGrid (5/19, vLLM 0.10.1.dev verified)
- [x] M8.T7: HuggingFace model weight cache (5/21, 95GB total across 3 models)
- [x] M8.T8: Repo clone to OG via PAT (5/21)
- [x] M8.T8b: Claude Code SIF deep-dive (5/21)
- [x] M8.T9: Adapt batch scripts for OrangeGrid (5/21, test_job_og.sub + wrapper.sh modifications)
- [ ] M8.T10: 100-text Llama test (5/21, in flight, ~14:43 EDT completion)
- [ ] M8.T10b: 100-text Llama with system/user prompt split (DEFERRED until V1 optimization decision)
- [ ] M8.T10c: V1 engine experiment (remove fp8 KV cache, re-measure RPM; expected 30-60 min)
- [ ] M8.T10d: Apophis re-baseline measurement (verify 195 RPM is API-call rate not row rate)
- [ ] M8.T10e: config2 vs config3 prompt-token impact (re-run test on config2, compare RPM)
- [ ] M8.T10f: Concurrent thread scaling (test 32 and 64 threads against current 16)
- [ ] M8.T11: 1,000-text characterization wave (3 models in parallel)
- [ ] M8.T12: Full 56K production runs (12 chunks per model, subset-based per Introne coordination)
- [ ] M8.T13: Merge chunk results locally on Apophis
- [ ] M8.T14: Statistical significance tests at full corpus scale
- [x] M8.T15: First timing report to Introne (in Production_Runtime_Estimates_Rev2.docx, pending send)
- [ ] M8.T16: Ongoing coordination pings during production wave

---

## OrangeGrid Submit File Reference

Add the following as a new reference section in WORKPLAN, since these gotchas are now confirmed-in-production and not documented in HPC_CAMEL_README.md.

### Canonical OG A100 Submit File Recipe

Required ClassAds for jobs targeting NVIDIA A100 80GB on OrangeGrid:

```
request_gpus    = N
+request_gpus   = N
+wantsArgusNode = True
request_cpus    = 8
request_memory  = 64GB
should_transfer_files = NO
Requirements = (CUDADeviceName == "NVIDIA A100 80GB PCIe") && (TARGET.GPUs >= 2)
environment = "HOME=/home/szkhan CAMEL_TP=1 CAMEL_CONFIG=config3.yaml HF_HOME=/home/szkhan/.cache/huggingface"
```

Critical: do NOT add `when_to_transfer_output = ON_EXIT` (contradicts ShouldTransferFiles=NO and errors at dry-run).

### Wrapper.sh Required Modifications

For OG compatibility, wrapper.sh needs three changes versus the Apophis template:

1. Accept `CAMEL_CONFIG` env var override: `CONFIG_FILE="${CAMEL_CONFIG:-config2.yaml}"`
2. Convert CUDA_VISIBLE_DEVICES UUIDs to integer indices before invoking Python:
```bash
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "Converted CUDA_VISIBLE_DEVICES UUIDs to integer indices: ${CUDA_VISIBLE_DEVICES}"
fi
```
3. Be executable: `chmod +x wrapper.sh` (git clone does not preserve executable bit in some configurations)

### Repository Directory Symlinks

Apophis repo has `chunks/`, `results/`, `logs/` as symlinks to `/DATA/szkhan/camel/hpc_data/...`. On OrangeGrid where `/DATA` does not exist, these arrive as broken symlinks and must be replaced with real directories: `rm chunks results logs && mkdir chunks results logs`.

---

## Cluster Topology Reference Table Update

Update the existing cluster table to capture the V0/V1 engine distinction:

| Attribute             | Apophis                           | OrangeGrid                              |
|-----------------------|-----------------------------------|-----------------------------------------|
| vLLM version          | 0.16.0 (pinned)                   | 0.10.1.dev1+gbcc0a3cbe (commit bcc0a3c) |
| vLLM engine           | V1 (default)                      | V0 (forced by fp8 KV cache; V1 unavailable in 0.10.1.dev with that flag) |
| CUDA driver           | 12.x                              | 12.8 (caps vLLM at 0.10.x)              |
| GPU config            | 2× A6000, TP=2                    | 1× A100 80GB, TP=1                      |
| Llama 3.3 RPM         | 195 (API-call rate to be verified) | 65 observed at V0 (TBD at V1)           |

---

## Production Estimates Reference

Replace any earlier wall-clock numbers in WORKPLAN with the corrected Rev 2 numbers:

| Scale                          | API calls  | OG Wall-clock at 65 RPM |
|--------------------------------|-----------|--------------------------|
| 100 texts (test)               | 7,500     | ~2 hours                 |
| 1,000 texts                    | 75,000    | ~19 hours                |
| 5,000 texts (1 production chunk) | 375,000 | ~96 hours (4 days)       |
| 56,796 texts (1 model, 12 parallel chunks) | 4.26M | ~4 days |
| Full 3-model corpus (parallel) | 12.78M    | 8 to 14 days (contention-dependent) |

If V1 engine optimization succeeds (~2x throughput): halve all of these.

---

## Side Items

- [x] DGX Spark research doc resent to Introne (done 5/14)
- [ ] Send Production_Runtime_Estimates_Rev2.docx to Introne (after final test numbers locked)
- [ ] Submit V1 engine optimization decision to Introne with revised wall-clock estimates
