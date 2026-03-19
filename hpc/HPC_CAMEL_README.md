# CAMEL Annotation — HPC Batch Processing

Self-contained scripts for running CAMEL annotation on HPC clusters using
HTCondor and vLLM. Adapted from Prof. Introne's belief extraction HPC pipeline.

## Overview

| File | Purpose |
|------|---------|
| `camel_annotate_hpc.py` | Main annotation script (starts vLLM, processes chunk, saves Feather) |
| `wrapper.sh` | Bash wrapper — sets up conda env and runs the script |
| `test_job.sub` | HTCondor submit file for test job (validates setup) |
| `split_data.py` | Splits corpus into chunks (reuse from project root) |
| `merge_results.py` | Merges chunk results into single Feather file |
| `config2.yaml` | CAMEL config with prompts, labels, guidelines (copy from project root) |

## Directory Structure

```
src/hpc/camel_annotation/
├── camel_annotate_hpc.py
├── wrapper.sh
├── test_job.sub
├── merge_results.py
├── config2.yaml          # copied from project root
├── split_data.py         # copied from project root
├── vllm-openai.sif       # Singularity container (pulled once)
├── chunks/               # created by split_data.py
│   ├── test_chunk.feather
│   ├── chunk_000.feather
│   └── ...
├── results/              # job outputs go here
│   └── ...
└── logs/                 # HTCondor logs
    └── ...
```

## Prerequisites

### 1. Singularity Container

Pull once on the login node:
```bash
singularity pull vllm-openai.sif docker://vllm/vllm-openai:v0.10.0
```

### 2. Conda Environment

```bash
conda create -n camel-annotation python=3.11 pandas pyarrow pyyaml requests
conda activate camel-annotation
```

### 3. Pre-download Model Weights

To avoid download failures during jobs:
```bash
huggingface-cli download casperhansen/llama-3.3-70b-instruct-awq
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ
huggingface-cli download Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ
```

### 4. Create Directories

```bash
mkdir -p chunks results logs
```

### 5. Copy Config

```bash
cp ../../config2.yaml .
```

## Quick Start

### Step 1: Prepare Corpus as Feather

If your corpus is CSV:
```python
import pandas as pd
df = pd.read_csv("path/to/full_corpus.csv")
df.to_feather("corpus.feather")
```

### Step 2: Create Test Chunk

```bash
python split_data.py --input corpus.feather --create-test --test-size 100
```

### Step 3: Run Test Job

```bash
condor_submit test_job.sub
```

Monitor:
```bash
condor_q $USER
tail -f logs/test_job.out
```

### Step 4: Verify Test Results

```python
import pandas as pd
df = pd.read_feather("results/test_results.feather")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(df[["text_id", "prompt_name", "model", "Care", "Honesty"]].head())
```

### Step 5: Split Production Data

```bash
python split_data.py --input corpus.feather --chunk-size 250000
```

This creates:
- `chunks/chunk_000.feather`, `chunk_001.feather`, ...
- `chunks/chunk_manifest.json`
- `batch_job.sub` (auto-generated HTCondor submit)

### Step 6: Run Production Batch

```bash
condor_submit batch_job.sub
```

Monitor:
```bash
condor_q $USER
watch -n 60 'ls -la results/*.feather | wc -l'
```

### Step 7: Merge Results

After all jobs complete:
```bash
# Check completeness first
python merge_results.py --check

# Merge
python merge_results.py --output camel_results_full.feather
```

### Step 8: Evaluate

```bash
# Copy merged results back to project and evaluate
python evaluate.py --results camel_results_full.feather --gold samples/50_random_samples_ans.csv
```

## Running Multiple Models

Each model gets a separate batch. Set model via environment variables in
`wrapper.sh` or submit separate jobs:

```bash
# Llama 3.3 70B (default)
condor_submit batch_job.sub

# Qwen 2.5 72B — edit wrapper.sh or use env override:
CAMEL_MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ" condor_submit batch_job.sub

# DeepSeek R1 32B (auto-detects higher max_tokens)
CAMEL_MODEL="Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ" condor_submit batch_job.sub
```

Or create per-model submit files for cleaner job management.

## Resource Requirements

| Model Size | GPUs | GPU VRAM | CUDA | CPU | RAM |
|-----------|------|----------|------|-----|-----|
| 70B (Llama, Qwen, AceGPT) | 2 | ≥40GB each | ≥12.8 | 8 | 32GB |
| 32B (DeepSeek, Falcon-H1) | 2 | ≥40GB each | ≥12.8 | 8 | 32GB |
| 24B (Mistral Small) | 1 | ≥24GB | ≥12.8 | 8 | 32GB |

## Output Format

Results are Feather files in wide format (one row per text × prompt):

| Column | Description |
|--------|-------------|
| `text_id` | Original text identifier |
| `prompt_id` | Prompt strategy ID |
| `prompt_name` | Prompt strategy name |
| `model` | Model identifier |
| `original_index` | Index in original corpus (for merging) |
| `Care`, `Honesty`, ... | 25 label columns (1=YES, 0=NO, -1=error) |
| `response__Care`, ... | 25 response text columns |

## Troubleshooting

### vLLM server doesn't start
```bash
cat logs/<job_name>.err
cat vllm_server.log  # in working dir
```

### Jobs stuck in queue
```bash
condor_status -constraint 'TotalGpus >= 2' -af Machine CUDADriverVersion CUDAGlobalMemoryMb
```

### Check job completeness
```bash
python merge_results.py --check
```
