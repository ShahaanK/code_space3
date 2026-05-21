#!/usr/bin/env python3
"""
CAMEL Annotation — Generate HPC Batch Submit Files
====================================================
Generates per-model HTCondor submit files with GPU requirements
routed by model size AND cluster target.

Two cluster targets (--cluster flag):

  apophis (default):
    70B models → 2 GPUs (TP=2), >=40GB VRAM each (RTX A6000)
    32B/24B   → 1 GPU,  >=24GB VRAM
    Conda env, no transfer_executable, request_disk=20GB

  orangegrid:
    70B models → 1 GPU (TP=1), A100 80GB targeted by device name
    32B/24B   → 1 GPU (TP=1), any CUDA 12+ GPU >=40GB VRAM
    Conda env, transfer_executable=false, no request_disk (auto)

CRITICAL: Never apply orangegrid config to Apophis or vice versa.
The 4/30 incident broke the validated Apophis TP=2 config by
applying OrangeGrid-style single-GPU settings.

Reads chunk manifest from split_data.py and generates:
  - batch_<model_short_name>.sub for each model
  - wrapper_<model_short_name>.sh with model baked in

Usage:
    # Apophis (default, TP=2 for 70B)
    python generate_batch_jobs.py --model llama-3.3-70b

    # OrangeGrid (TP=1, A100 targeting)
    python generate_batch_jobs.py --cluster orangegrid --model llama-3.3-70b

    # List models with cluster-specific GPU assignments
    python generate_batch_jobs.py --cluster orangegrid --list-models

Then submit:
    condor_submit batch_llama-3.3-70b.sub

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path


# =============================================================================
# MODEL REGISTRY — Apophis baseline (TP=2 for 70B, validated March 2026)
# =============================================================================

MODELS = [
    {
        "short_name": "llama-3.3-70b",
        "hf_name": "casperhansen/llama-3.3-70b-instruct-awq",
        "origin": "Western (Meta, USA)",
        "quantization": "awq_marlin",
        "tp": 2,
        "request_gpus": 2,
        "gpu_vram_mb": 40000,
        "max_model_len": 2048,
        "max_tokens": 256,
        "extra_flags": "",
    },
    {
        "short_name": "qwen2.5-72b",
        "hf_name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "origin": "Eastern (Alibaba, China)",
        "quantization": "awq_marlin",
        "tp": 2,
        "request_gpus": 2,
        "gpu_vram_mb": 40000,
        "max_model_len": 2048,
        "max_tokens": 256,
        "extra_flags": "--trust-remote-code",
    },
    {
        "short_name": "deepseek-r1-32b",
        "hf_name": "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
        "origin": "Eastern (DeepSeek, China)",
        "quantization": "awq_marlin",
        "tp": 1,
        "request_gpus": 1,
        "gpu_vram_mb": 24000,
        "max_model_len": 4096,
        "max_tokens": 1024,
        "extra_flags": "--trust-remote-code",
    },
    {
        "short_name": "mistral-small-24b",
        "hf_name": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
        "origin": "Western (Mistral, France)",
        "quantization": "compressed-tensors",
        "tp": 1,
        "request_gpus": 1,
        "gpu_vram_mb": 24000,
        "max_model_len": 2048,
        "max_tokens": 256,
        "extra_flags": "",
    },
    {
        "short_name": "acegpt-70b",
        "hf_name": "FreedomIntelligence/AceGPT-v2-70B-chat",
        "origin": "Arabic (KAUST/UAE)",
        "quantization": "awq_marlin",
        "tp": 2,
        "request_gpus": 2,
        "gpu_vram_mb": 40000,
        "max_model_len": 2048,
        "max_tokens": 256,
        "extra_flags": "--trust-remote-code",
    },
    {
        "short_name": "falcon-h1-34b",
        "hf_name": "tiiuae/Falcon-H1-34B-Instruct",
        "origin": "Arabic/Gulf (TII, UAE)",
        "quantization": "awq_marlin",
        "tp": 1,
        "request_gpus": 1,
        "gpu_vram_mb": 24000,
        "max_model_len": 2048,
        "max_tokens": 256,
        "extra_flags": "--trust-remote-code",
    },
]


# =============================================================================
# ORANGEGRID OVERRIDES — TP=1 on A100 80GB (locked 5/11/2026)
# =============================================================================
# Strategy: single A100 80GB per 70B/72B model. 37GB weights + 43GB headroom.
# 32B/24B models fit on any CUDA 12+ GPU (A100, L40S, A40 all have >=45GB).
# See WORKLOG 5/11 for full rationale.
# =============================================================================

ORANGEGRID_OVERRIDES = {
    # 70B/72B: TP=1 single A100 80GB (targeted by device name)
    "llama-3.3-70b":    {"tp": 1, "request_gpus": 1, "gpu_vram_mb": 70000},
    "qwen2.5-72b":      {"tp": 1, "request_gpus": 1, "gpu_vram_mb": 70000},
    "acegpt-70b":       {"tp": 1, "request_gpus": 1, "gpu_vram_mb": 70000},
    # 32B/24B: already TP=1, keep broad targeting
    "deepseek-r1-32b":  {"tp": 1, "request_gpus": 1, "gpu_vram_mb": 40000},
    "mistral-small-24b": {"tp": 1, "request_gpus": 1, "gpu_vram_mb": 40000},
    "falcon-h1-34b":    {"tp": 1, "request_gpus": 1, "gpu_vram_mb": 40000},
}

# Models that require A100 80GB (too large for L40S/A40 45GB with headroom)
A100_REQUIRED_MODELS = {"llama-3.3-70b", "qwen2.5-72b", "acegpt-70b"}


# =============================================================================
# CLUSTER CONFIGURATIONS
# =============================================================================

CLUSTER_CONFIGS = {
    "apophis": {
        "description": "Prof. Introne's lab box (2x RTX A6000, no scheduler)",
        "conda_path": "${HOME}/miniconda3",
        "conda_env": "camel-annotation",
        "container": "${HOME}/vllm-openai.sif",
        "use_request_disk": True,
        "request_disk": "20GB",
        "transfer_executable": True,
        # Apophis uses CUDADriverVersion >= 12.8 (both A6000s same driver)
        "min_cuda_driver": 12.8,
    },
    "orangegrid": {
        "description": "Syracuse ITS HTC cluster (HTCondor, A100/L40S/A40)",
        "conda_path": "${HOME}/miniconda3",
        "conda_env": "camel-annotation",
        "container": "${HOME}/vllm-openai.sif",
        "use_request_disk": False,   # JOB_DEFAULT_REQUESTDISK=DiskUsage (auto)
        "request_disk": None,
        "transfer_executable": False,  # NFS shared filesystem
        # OrangeGrid: A100=12.8, A40=12.0, L40S=12.8. Floor at 12.0.
        "min_cuda_driver": 12.0,
    },
}


# Default paths (edit these or override via args)
DEFAULT_CONDA_PATH = "${HOME}/miniconda3"
DEFAULT_CONDA_ENV = "camel-annotation"
DEFAULT_CONTAINER = "${HOME}/vllm-openai.sif"
DEFAULT_CONFIG = "config2.yaml"
DEFAULT_THREADS = 16
DEFAULT_PORT = 8900


def get_model(name):
    """Look up model by short_name or hf_name."""
    for m in MODELS:
        if name in (m["short_name"], m["hf_name"]):
            return m
    return None


def apply_cluster_overrides(models, cluster):
    """Return a new model list with cluster-specific overrides applied."""
    if cluster == "apophis":
        return models  # Apophis uses baseline registry as-is

    if cluster == "orangegrid":
        result = []
        for m in models:
            m_copy = copy.deepcopy(m)
            overrides = ORANGEGRID_OVERRIDES.get(m["short_name"], {})
            m_copy.update(overrides)
            result.append(m_copy)
        return result

    raise ValueError(f"Unknown cluster: {cluster}")


def load_manifest(chunks_dir):
    """Load chunk manifest from split_data.py output."""
    manifest_path = Path(chunks_dir) / "chunk_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def generate_wrapper(model, script_dir, conda_path, conda_env,
                     container, config, threads, port):
    """Generate a model-specific wrapper.sh."""
    name = model["short_name"]
    wrapper_path = script_dir / f"wrapper_{name}.sh"

    content = f"""#!/bin/bash
# =============================================================================
# CAMEL Annotation — HTCondor Wrapper for {name}
# =============================================================================
# Model: {model['hf_name']}
# Origin: {model['origin']}
# GPUs: {model['request_gpus']} (TP={model['tp']})
# Auto-generated by generate_batch_jobs.py
# =============================================================================

set -euo pipefail

CHUNK_FILE="$1"
RESULT_FILE="$2"

echo "============================================================"
echo "CAMEL Annotation — {name}"
echo "============================================================"
echo "  Chunk:    ${{CHUNK_FILE}}"
echo "  Output:   ${{RESULT_FILE}}"
echo "  Model:    {model['hf_name']}"
echo "  GPUs:     {model['request_gpus']} (TP={model['tp']})"
echo "  Host:     $(hostname)"
echo "  Date:     $(date)"
echo "============================================================"

# Setup conda
source "{conda_path}/etc/profile.d/conda.sh"
conda activate "{conda_env}"

# Verify files
for f in "${{CHUNK_FILE}}" "{container}" "{config}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

# Run annotation
python camel_annotate_hpc.py \\
    "${{CHUNK_FILE}}" \\
    "${{RESULT_FILE}}" \\
    --model "{model['hf_name']}" \\
    --config "{config}" \\
    --container "{container}" \\
    --quantization "{model['quantization']}" \\
    --tp {model['tp']} \\
    --threads {threads} \\
    --port {port}

echo "Job finished with exit code: $?"
echo "Date: $(date)"
"""

    with open(wrapper_path, "w") as f:
        f.write(content)
    os.chmod(wrapper_path, 0o755)

    return wrapper_path


def generate_submit(model, chunks_dir, results_subdir, script_dir,
                    manifest, cluster_config, cluster):
    """Generate a model-specific HTCondor submit file."""
    name = model["short_name"]
    submit_path = script_dir / f"batch_{name}.sub"
    chunks_subdir = Path(chunks_dir).name

    # Build queue list from manifest
    queue_items = []
    for chunk in manifest["chunks"]:
        chunk_file = f"{chunks_subdir}/{chunk['filename']}"
        result_file = f"{results_subdir}_{name}/results_{chunk['chunk_id']:03d}.feather"
        job_name = f"{name}_chunk_{chunk['chunk_id']:03d}"
        queue_items.append(f"    {chunk_file}, {result_file}, {job_name}")

    queue_list = "\n".join(queue_items)

    # GPU requirements based on model size AND cluster
    gpus = model["request_gpus"]
    vram = model["gpu_vram_mb"]
    min_driver = cluster_config["min_cuda_driver"]

    if cluster == "orangegrid":
        requirements = _orangegrid_requirements(model, min_driver)
    else:
        requirements = _apophis_requirements(model, min_driver)

    # Resource lines (cluster-specific)
    resource_lines = f"""request_gpus   = {gpus}
+request_gpus  = {gpus}
request_cpus   = 8
request_memory = 32GB"""

    if cluster_config["use_request_disk"]:
        resource_lines += f"\nrequest_disk   = {cluster_config['request_disk']}"

    # Transfer executable
    transfer_line = ""
    if not cluster_config["transfer_executable"]:
        transfer_line = "\ntransfer_executable = false"

    content = f"""#!/usr/bin/env condor_submit
# =============================================================================
# CAMEL Annotation — BATCH JOB: {name}
# =============================================================================
# Model:   {model['hf_name']}
# Origin:  {model['origin']}
# Cluster: {cluster}
# GPUs:    {gpus} (TP={model['tp']})
# VRAM:    >= {vram} MB per GPU
# Chunks:  {manifest['total_chunks']}
#
# Auto-generated by generate_batch_jobs.py --cluster {cluster}
#
# Usage:
#   condor_submit batch_{name}.sub
#
# Monitor:
#   condor_q $USER
#   watch -n 30 'ls -la {results_subdir}_{name}/*.feather | wc -l'
# =============================================================================

initialdir   = {script_dir.resolve()}

executable   = wrapper_{name}.sh
arguments    = $(chunk_file) $(result_file){transfer_line}

output       = logs/$(job_name).out
error        = logs/$(job_name).err
log          = logs/$(job_name).log

# Resource requirements — {name} ({gpus} GPU{'s' if gpus > 1 else ''}, {cluster})
{resource_lines}

Requirements = {requirements}

# Environment
environment = "HF_HOME=$ENV(HOME)/.cache/huggingface"

# Retry failed jobs once
on_exit_hold = (ExitCode != 0)
periodic_release = (NumJobStarts < 2) && ((CurrentTime - EnteredCurrentStatus) > 300)

# Queue one job per chunk
queue chunk_file, result_file, job_name from (
{queue_list}
)
"""

    with open(submit_path, "w") as f:
        f.write(content)

    return submit_path


def _apophis_requirements(model, min_driver):
    """Build Condor requirements expression for Apophis."""
    gpus = model["request_gpus"]
    vram = model["gpu_vram_mb"]

    if gpus >= 2:
        return (
            f"(CUDAGlobalMemoryMb >= {vram}) && "
            f"(CUDADriverVersion >= {min_driver}) && "
            f"(TotalGpus >= {gpus})"
        )
    else:
        return (
            f"(CUDAGlobalMemoryMb >= {vram}) && "
            f"(CUDADriverVersion >= {min_driver})"
        )


def _orangegrid_requirements(model, min_driver):
    """
    Build Condor requirements expression for OrangeGrid.

    OrangeGrid pool (5/11 recon, post-ITS visibility expansion):
      9x A100 80GB PCIe (2 GPUs each, driver 12.8)
      7x A40           (4 GPUs each, 45GB, driver 12.0)
     11x L40S          (2 GPUs each, 45GB, driver 12.8)

    Condor 9.8 uses CUDA-prefixed ClassAd attributes:
      CUDADeviceName, CUDAGlobalMemoryMb, CUDADriverVersion

    70B/72B models (37GB weights): must land on A100 80GB for headroom.
    32B/24B models (17-18GB weights): any CUDA 12+ GPU with >=40GB works.
    """
    name = model["short_name"]

    if name in A100_REQUIRED_MODELS:
        # Target A100 80GB by device name (most reliable on heterogeneous pool)
        return (
            f'(CUDADeviceName == "NVIDIA A100 80GB PCIe") && '
            f"(CUDADriverVersion >= {min_driver})"
        )
    else:
        # 32B/24B: any CUDA 12+ GPU with enough VRAM
        vram = model["gpu_vram_mb"]
        return (
            f"(CUDAGlobalMemoryMb >= {vram}) && "
            f"(CUDADriverVersion >= {min_driver})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-model HTCondor batch submit files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cluster", choices=["apophis", "orangegrid"],
                        default="apophis",
                        help="Target cluster (default: apophis). "
                             "Controls TP, GPU requirements, and submit template.")
    parser.add_argument("--model", type=str, default=None,
                        help="Generate for specific model only (short name). "
                             "Default: all models.")
    parser.add_argument("--chunks-dir", default="chunks",
                        help="Directory with chunk files and manifest")
    parser.add_argument("--results-dir", default="results",
                        help="Base results directory prefix (per-model subdirs created)")
    parser.add_argument("--conda-path", default=None,
                        help=f"Override conda path (default from cluster config)")
    parser.add_argument("--conda-env", default=None,
                        help=f"Override conda env (default from cluster config)")
    parser.add_argument("--container", default=None,
                        help=f"Override Singularity container path")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    args = parser.parse_args()

    # --- Cluster config ---
    cluster = args.cluster
    cluster_config = CLUSTER_CONFIGS[cluster]

    # Resolve paths: CLI overrides > cluster config > defaults
    conda_path = args.conda_path or cluster_config["conda_path"]
    conda_env = args.conda_env or cluster_config["conda_env"]
    container = args.container or cluster_config["container"]

    # --- Apply cluster overrides to model registry ---
    models = apply_cluster_overrides(MODELS, cluster)

    # --- List models ---
    if args.list_models:
        print(f"\nCluster: {cluster} ({cluster_config['description']})")
        print(f"\n{'Short Name':<22s} {'GPUs':>4s} {'TP':>3s} {'VRAM':>7s}  {'HuggingFace Name'}")
        print("-" * 90)
        for m in models:
            print(f"{m['short_name']:<22s} {m['request_gpus']:>4d} {m['tp']:>3d} "
                  f"{m['gpu_vram_mb']:>5d}MB  {m['hf_name']}")
        return

    # --- Load manifest ---
    manifest = load_manifest(args.chunks_dir)
    if not manifest:
        print(f"ERROR: No chunk_manifest.json found in {args.chunks_dir}/")
        print(f"  Run split_data.py first to create chunks.")
        sys.exit(1)

    print("=" * 70)
    print(f"CAMEL Annotation — Generate HPC Batch Jobs [{cluster}]")
    print("=" * 70)
    print(f"  Cluster:     {cluster} ({cluster_config['description']})")
    print(f"  Chunks dir:  {args.chunks_dir}")
    print(f"  Chunks:      {manifest['total_chunks']}")
    print(f"  Chunk size:  {manifest['chunk_size']}")
    print(f"  Total texts: {manifest['total_unprocessed']}")

    # --- Select models ---
    if args.model:
        model = None
        for m in models:
            if args.model in (m["short_name"], m["hf_name"]):
                model = m
                break
        if not model:
            print(f"\nERROR: Unknown model '{args.model}'")
            print(f"  Available: {', '.join(m['short_name'] for m in models)}")
            sys.exit(1)
        models_to_generate = [model]
    else:
        models_to_generate = models

    script_dir = Path(".")

    # Create logs dir
    Path("logs").mkdir(exist_ok=True)

    print(f"\n  Generating submit files for {len(models_to_generate)} model(s):\n")

    for model in models_to_generate:
        name = model["short_name"]
        gpus = model["request_gpus"]

        # Create per-model results directory
        results_dir = f"{args.results_dir}_{name}"
        Path(results_dir).mkdir(exist_ok=True)

        # Generate wrapper
        wrapper_path = generate_wrapper(
            model=model,
            script_dir=script_dir,
            conda_path=conda_path,
            conda_env=conda_env,
            container=container,
            config=args.config,
            threads=args.threads,
            port=args.port,
        )

        # Generate submit file
        submit_path = generate_submit(
            model=model,
            chunks_dir=args.chunks_dir,
            results_subdir=args.results_dir,
            script_dir=script_dir,
            manifest=manifest,
            cluster_config=cluster_config,
            cluster=cluster,
        )

        # Describe node targeting
        if cluster == "orangegrid" and name in A100_REQUIRED_MODELS:
            node_type = "A100 80GB only (TP=1)"
        elif gpus >= 2:
            node_type = "2-GPU high-VRAM nodes"
        else:
            node_type = "any CUDA 12+ node with 1 GPU"
        print(f"  {name:<22s} -> {gpus} GPU{'s' if gpus > 1 else ' '}  "
              f"({node_type})")
        print(f"    Submit:  {submit_path}")
        print(f"    Wrapper: {wrapper_path}")
        print(f"    Results: {results_dir}/")
        print()

    # --- Summary ---
    print("=" * 70)
    print(f"READY TO SUBMIT [{cluster}]")
    print("=" * 70)
    print()
    for model in models_to_generate:
        name = model["short_name"]
        print(f"  condor_submit batch_{name}.sub")
    print()
    print("Monitor:")
    print("  condor_q $USER")
    if cluster == "orangegrid":
        print("  condor_q -better-analyze $USER    # if jobs idle")
    print()
    print("After all jobs complete, merge per-model:")
    for model in models_to_generate:
        name = model["short_name"]
        print(f"  python merge_results.py "
              f"--results-dir results_{name} "
              f"--output merged_{name}.feather")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
