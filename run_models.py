#!/usr/bin/env python3
"""
Model Runner — Automated Multi-Model Pipeline for Apophis
==========================================================
Cycles through multiple models on vLLM without manual intervention:
  1. Starts vLLM with model N
  2. Waits for server to be ready
  3. Runs the annotation pipeline
  4. Kills vLLM and verifies GPU memory freed
  5. Moves to model N+1

Usage:
    python run_models.py                           # run all enabled models
    python run_models.py --dry-run                 # show what would run
    python run_models.py --models 0,2              # run specific models by index
    python run_models.py --config config2.yaml     # alternate config
    python run_models.py --workers 16              # worker count for pipeline

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import yaml


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
# Each entry defines how to start vLLM for that model. These are separate from
# config.yaml because vLLM start parameters (quantization, TP size) differ
# per model and don't belong in the annotation config.
# =============================================================================

LOCAL_MODELS = [
    {
        "name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "short_name": "llama-3.1-70b",
        "origin": "Western (Meta, USA)",
        "quantization": "awq_marlin",
        "tensor_parallel_size": 2,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.95,
        "extra_flags": "",
        "enabled": True,
    },
    {
        "name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "short_name": "qwen2.5-72b",
        "origin": "Eastern (Alibaba, China)",
        "quantization": "awq_marlin",
        "quantization_fallback": "gptq_marlin",
        "quantization_fallback_model": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        "tensor_parallel_size": 2,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.95,
        "extra_flags": "--trust-remote-code",
        "enabled": True,
    },
    {
        "name": "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
        "short_name": "deepseek-r1-32b",
        "origin": "Eastern (DeepSeek, China)",
        "quantization": "awq_marlin",
        "quantization_fallback": "awq",
        "tensor_parallel_size": 2,
        "max_model_len": 4096,  # R1 reasoning can be longer
        "gpu_memory_utilization": 0.95,
        "extra_flags": "--trust-remote-code",
        "enabled": True,
    },
    {
        "name": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
        "short_name": "mistral-small-24b",
        "origin": "Western (Mistral, France)",
        "quantization": "compressed-tensors",
        "tensor_parallel_size": 2,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.95,
        "extra_flags": "",
        "enabled": False,  # enable when ready to test
    },
    {
        "name": "FreedomIntelligence/AceGPT-v2-70B-chat",
        "short_name": "acegpt-70b",
        "origin": "Arabic (KAUST/UAE)",
        "quantization": "awq_marlin",
        "quantization_fallback": "awq",
        "tensor_parallel_size": 2,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.95,
        "extra_flags": "--trust-remote-code",
        "enabled": False,  # enable when ready to test
    },
]


# =============================================================================
# VLLM MANAGEMENT
# =============================================================================

VLLM_PORT = 8900
VLLM_API_KEY = "sk-local-shahaan"
VLLM_ENV = os.path.expanduser("~/.venv-vllm/bin/activate")
PIPELINE_ENV = os.path.expanduser("~/.venv-camel/bin/activate")


def start_vllm(model_config, log_file="vllm_server.log"):
    """Start vLLM server as a background process. Returns the process."""
    name = model_config["name"]
    quant = model_config["quantization"]
    tp = model_config["tensor_parallel_size"]
    gpu_util = model_config["gpu_memory_utilization"]
    max_len = model_config.get("max_model_len", 2048)
    extra = model_config.get("extra_flags", "")

    cmd = (
        f"source {VLLM_ENV} && "
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {name} "
        f"--quantization {quant} "
        f"--tensor-parallel-size {tp} "
        f"--gpu-memory-utilization {gpu_util} "
        f"--max-model-len {max_len} "
        f"--kv-cache-dtype fp8_e4m3 "
        f"--num-scheduler-steps 5 "
        f"--port {VLLM_PORT} "
        f"--api-key \"{VLLM_API_KEY}\""
    )
    if extra:
        cmd += f" {extra}"

    print(f"    Starting vLLM: {name}")
    print(f"    Quantization: {quant}, TP: {tp}, GPU util: {gpu_util}")
    print(f"    Max model len: {max_len}, KV cache: fp8_e4m3")

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            cmd, shell=True, executable="/bin/bash",
            stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # create process group for clean kill
        )

    return proc


def wait_for_vllm(timeout=600, poll_interval=10):
    """Wait for vLLM to be ready by polling the /v1/models endpoint."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{VLLM_PORT}/v1/models"
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    start = time.time()

    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            if data.get("data"):
                model_id = data["data"][0]["id"]
                elapsed = time.time() - start
                print(f"    vLLM ready in {elapsed:.0f}s (model: {model_id})")
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, Exception):
            pass

        time.sleep(poll_interval)

    print(f"    ERROR: vLLM did not start within {timeout}s")
    return False


def kill_vllm(proc):
    """Kill the vLLM process group and verify GPU memory is freed."""
    if proc and proc.poll() is None:
        print("    Stopping vLLM...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
            except Exception:
                pass

    # Wait for GPU memory to be freed
    print("    Waiting for GPU memory to free...", end=" ", flush=True)
    for _ in range(30):
        time.sleep(2)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            mem_values = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            # Both GPUs should be under 1000 MB when idle
            if all(m < 1000 for m in mem_values):
                print(f"clear ({mem_values} MB)")
                return True
        except Exception:
            pass

    print("WARNING: GPU memory may not be fully freed")
    return False


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def update_config_model(config_path, model_name):
    """Update the config file to enable only the specified model."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for m in config["models"]:
        if m.get("provider") == "local_vllm":
            m["enabled"] = (m["name"] == model_name)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                  width=120, allow_unicode=True)

    return config


def run_pipeline(config_path, workers, dry_run=False):
    """Run the annotation pipeline and return the exit code."""
    cmd = f"source {PIPELINE_ENV} && python run_annotation.py --config {config_path} --workers {workers}"
    if dry_run:
        cmd += " --dry-run"

    print(f"    Running pipeline (workers={workers})...")
    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash"
    )
    return result.returncode


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Runner for Apophis")
    parser.add_argument("--config", default="config2.yaml",
                        help="Config file to use")
    parser.add_argument("--workers", type=int, default=16,
                        help="Worker threads for pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model indices to run (e.g. 0,2,3)")
    parser.add_argument("--vllm-timeout", type=int, default=600,
                        help="Seconds to wait for vLLM startup (default: 600)")
    args = parser.parse_args()

    # Filter models
    if args.models:
        indices = [int(i) for i in args.models.split(",")]
        models_to_run = [LOCAL_MODELS[i] for i in indices if i < len(LOCAL_MODELS)]
    else:
        models_to_run = [m for m in LOCAL_MODELS if m.get("enabled", True)]

    if not models_to_run:
        print("ERROR: No models to run.")
        return

    # Print plan
    print("=" * 70)
    print("MODEL RUNNER \u2014 Automated Multi-Model Pipeline")
    print("=" * 70)
    print(f"  Config:  {args.config}")
    print(f"  Workers: {args.workers}")
    print(f"  Models:  {len(models_to_run)}")
    for i, m in enumerate(models_to_run):
        print(f"    [{i}] {m['short_name']:25s} ({m['origin']})")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)

    if args.dry_run:
        print("\n  *** DRY RUN \u2014 showing plan only ***\n")
        for i, model in enumerate(models_to_run):
            print(f"  Step {i+1}: Load {model['name']}")
            print(f"          Quant: {model['quantization']}, TP: {model['tensor_parallel_size']}")
            print(f"          Run pipeline with --workers {args.workers}")
            print(f"          Kill vLLM, free GPUs")
            print()
        print("  Use without --dry-run to execute.")
        return

    # Run each model
    results = []
    total_start = time.time()

    for i, model in enumerate(models_to_run):
        print(f"\n{'='*70}")
        print(f"MODEL {i+1}/{len(models_to_run)}: {model['short_name']}")
        print(f"{'='*70}")

        model_start = time.time()
        status = "SUCCESS"
        vllm_proc = None

        try:
            # Step 1: Update config
            print(f"\n  [1/4] Updating config to enable {model['name']}...")
            update_config_model(args.config, model["name"])

            # Step 2: Start vLLM
            print(f"\n  [2/4] Starting vLLM...")
            vllm_proc = start_vllm(model, log_file=f"vllm_{model['short_name']}.log")

            if not wait_for_vllm(timeout=args.vllm_timeout):
                # Try fallback quantization if available
                if model.get("quantization_fallback"):
                    fallback_quant = model["quantization_fallback"]
                    fallback_model_name = model.get("quantization_fallback_model", model["name"])
                    print(f"    Retrying with fallback: {fallback_model_name} ({fallback_quant})")
                    kill_vllm(vllm_proc)
                    fallback = dict(model)
                    fallback["quantization"] = fallback_quant
                    fallback["name"] = fallback_model_name
                    vllm_proc = start_vllm(fallback, log_file=f"vllm_{model['short_name']}_fallback.log")
                    if not wait_for_vllm(timeout=args.vllm_timeout):
                        status = "FAILED: vLLM startup (both primary and fallback)"
                        continue
                    # Update config to use the fallback model name
                    update_config_model(args.config, fallback_model_name)
                else:
                    status = "FAILED: vLLM startup"
                    continue

            # Step 3: Run pipeline
            print(f"\n  [3/4] Running annotation pipeline...")
            exit_code = run_pipeline(args.config, args.workers)
            if exit_code != 0:
                status = f"FAILED: pipeline exit code {exit_code}"

            # Step 4: Kill vLLM
            print(f"\n  [4/4] Shutting down vLLM...")
            kill_vllm(vllm_proc)
            vllm_proc = None

        except KeyboardInterrupt:
            print("\n\n  INTERRUPTED by user. Cleaning up...")
            if vllm_proc:
                kill_vllm(vllm_proc)
            sys.exit(1)

        except Exception as e:
            status = f"FAILED: {str(e)}"

        finally:
            if vllm_proc:
                kill_vllm(vllm_proc)

        elapsed = time.time() - model_start
        results.append({
            "model": model["short_name"],
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
        })

        print(f"\n  {model['short_name']}: {status} ({elapsed:.0f}s)")

        # Brief pause between models
        if i < len(models_to_run) - 1:
            print("\n  Pausing 15s before next model...")
            time.sleep(15)

    # === SUMMARY ===
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print("RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print()
    print(f"  {'Model':<25s} {'Status':<30s} {'Time':>10s}")
    print(f"  {'-'*65}")
    for r in results:
        marker = "\u2713" if r["status"] == "SUCCESS" else "\u2717"
        print(f"  {marker} {r['model']:<23s} {r['status']:<30s} {r['elapsed_seconds']:>8.0f}s")

    # Save summary
    summary_path = f"outputs/model_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"results": results, "total_seconds": round(total_elapsed, 1)}, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
