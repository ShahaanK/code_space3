# CAMEL LLM Annotation Pipeline

**Cultural and Moral Expressions in Language (CAMEL) — LLM Text Annotation Benchmarking**

Automated pipeline for evaluating how different Large Language Models annotate
social media text for 25 psychological and cultural constructs (moral foundations,
cultural dimensions, behavioral markers) using the CAMEL annotation framework.

## Research Team

- **Shahaan Khan** — MS Applied Human-Centered AI, Syracuse University iSchool
- **Prof. Joshua Introne** — Advisor, Syracuse University iSchool
- **Prof. Mohammad Atari** — Collaborator, UMass Amherst (Culture and Morality Lab)

## Pipeline Architecture

The pipeline uses a config-driven, modular design:

| File | Purpose |
|------|---------|
| `config2.yaml` | Master config: prompts, labels, model specs, runtime params |
| `prompt_builder.py` | Builds annotation prompts with system/user split for prefix caching |
| `adapter.py` | OpenAI-compatible API client (sync/async, any provider) |
| `run_annotation.py` | Main annotation runner (threaded, checkpoint/resume, Feather output) |
| `run_models.py` | Multi-model orchestrator (starts vLLM, runs annotation, rotates models) |
| `evaluate.py` | Scoring: F1-macro/micro, precision, recall per label/model/prompt |
| `benchmark_workers.py` | Worker count throughput sweep |
| `update_config_models.py` | Syncs config model entries with LOCAL_MODELS dict |

## Models

| Model | Origin | Size | Role |
|-------|--------|------|------|
| Llama 3.3 70B | Western (Meta) | 70B | Western baseline |
| Qwen 2.5 72B | Eastern (Alibaba) | 72B | Eastern baseline |
| DeepSeek R1 32B | Eastern (DeepSeek) | 32B | Reasoning architecture |

## Data

- **Corpus:** 56,796 social media texts annotated by human coders (CAMEL V1)
- **Labels:** 25 constructs across moral foundations, cultural dimensions, and behavioral markers
- **Gold standard:** Human annotator majority voting (2+ of 3 agree)
- **Data files** are stored on `/DATA` (not in this repo) — see `samples/` symlink

## Directories

| Directory | Contents |
|-----------|----------|
| `samples/` | Corpus files, answer keys (symlink to /DATA) |
| `outputs/` | Run results, eval reports (symlink to /DATA) |
| `hpc/` | HTCondor batch processing scripts for HPC cluster |

## Quick Start
```bash
# Activate environment
source ~/myenv/bin/activate

# Start vLLM (if not running)
# See run_models.py for automated startup

# Run annotation on sample
python run_annotation.py --config config2.yaml --run-name test_run

# Evaluate results
python evaluate.py --results outputs/run_NNN_test_run/ --gold samples/sample_prompts_human_key.xlsx
```

## Project Documentation

- `WORKLOG.md` — Reverse-chronological development narrative
- `WORKPLAN.md` — Task tracking with milestones and changelog
