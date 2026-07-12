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
