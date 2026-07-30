# hpc/

HTCondor + vLLM batch processing for the CAMEL production 56K wave on
OrangeGrid. **See `HPC_CAMEL_README.md` for the full setup, quick-start, and
troubleshooting guide** — this file is just an orientation index.

## Key files

| File | Purpose |
|------|---------|
| `camel_annotate_hpc.py` | Self-contained per-chunk annotation runner (starts vLLM, calls it over urllib, saves Feather). Decoding params live in `MODEL_OVERRIDES` here, not in the config YAML. |
| `wrapper.sh` | Job-launch shell wrapper (CAMEL_CONFIG override, CUDA UUID→index conversion, HF offline env). |
| `batch_<model_tag>.sub` | Per-model production submit files (llama-3.3-70b, qwen2.5-72b, deepseek-r1-32b, mistral-small-24b, acegpt-70b, falcon-h1-34b) — run-versioned naming, subset-protocol chunk lists. **deepseek-r1-32b is launch-blocked and mistral-small-24b/acegpt-70b/falcon-h1-34b are unvalidated — do not submit them without passing each model's own chunk_000 gate first (falcon also needs a `--trust-remote-code` MODEL_OVERRIDES entry added to `camel_annotate_hpc.py` before it can run at all). See HPC_CAMEL_README.md and CLAUDE.md "Active Problems".** |
| `submit_wave.sh` | Submit entrypoint — consumes one global run number via `next_run.sh`, then `condor_submit`s a `batch_<tag>.sub` with `RUN_NO`/`SUBDATE` macros. |
| `next_run.sh` | Atomic (flock) global run-number counter consumer. Called once per human submit, never from inside a `.sub`. |
| `run_counter.txt` | Seed value for the run counter. The authoritative live counter is on OrangeGrid (the submit box) — never reseed it from this file. |
| `select_and_merge.py` | Apophis-side: selects one canonical run per chunk from the run-versioned result files (excludes crash-resume `.part_*.feather` shards) and concatenates to `merged_<tag>.feather` for `evaluate.py`. |
| `generate_batch_jobs.py`, `split_data.py` | Legacy/utility: per-model submit generation (superseded for production models by the hand-authored `batch_<tag>.sub` recipe), corpus chunking. |
| `config2.yaml`, `config3.yaml` | HPC copies of the master configs (`Number`/`text` column convention). `config3.yaml` is what OG production jobs read. |
| `chunklist_first.txt`, `chunklist_rest.txt` | Subset-protocol chunk lists (chunk_000 only, then 001..056) consumed by `submit_wave.sh`. |
| `chunks/`, `logs/`, `results/` | Symlinks to `/DATA/szkhan/camel/hpc_data/` on Apophis (real directories on OG). |
| `results_<model_tag>/` | Per-model run-versioned result Feather files, e.g. `results_qwen2.5-72b/`. |

For the run-versioned output naming schema, the submit flow, prerequisites,
quick-start steps, GPU routing, and troubleshooting — see
**`HPC_CAMEL_README.md`**.
