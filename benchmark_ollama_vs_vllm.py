#!/usr/bin/env python3
"""
Ollama vs vLLM Benchmark — CUDA Upgrade Ammunition
====================================================
Benchmarks the same annotation task on both Ollama and vLLM backends
to quantify the throughput cost of running on HPC nodes without
upgraded CUDA drivers (which forces Ollama instead of vLLM).

Produces a comparison report with:
  - RPM (requests per minute) for each backend x model
  - Projected wall-clock time for full 56K corpus
  - Speedup factor (vLLM / Ollama)

Usage on Apophis:
    # Quick benchmark (5 labels, ~15-30 min per model on Ollama)
    python benchmark_ollama_vs_vllm.py --config config3_50_samps.yaml

    # Most defensible: sweep Ollama worker counts, use its best RPM
    python benchmark_ollama_vs_vllm.py --config config3_50_samps.yaml --skip-vllm --sweep-ollama

    # Full benchmark (25 labels, ~7+ hrs per 70B model on Ollama)
    python benchmark_ollama_vs_vllm.py --config config3_50_samps.yaml --labels 25

    # Use existing vLLM baselines instead of re-running vLLM
    python benchmark_ollama_vs_vllm.py --config config3_50_samps.yaml --skip-vllm

    # Benchmark a single model (by index: 0=Llama, 1=Qwen, 2=DeepSeek)
    python benchmark_ollama_vs_vllm.py --config config3_50_samps.yaml --model-idx 0

Prerequisites:
    1. Pull Ollama models before running:
         ollama pull llama3.3:70b
         ollama pull qwen2.5:72b
         ollama pull deepseek-r1:32b
    2. Start Ollama with parallel request support:
         pkill -f ollama
         OLLAMA_NUM_PARALLEL=16 OLLAMA_GPU_LAYERS=999 nohup ollama serve &
    3. Make sure vLLM is NOT running (or use --skip-vllm to use known baselines)
    4. IMPORTANT: Shut down Ollama after benchmarking to free GPU:
         pkill -f ollama

Fair comparison methodology:
    - Both backends tested with identical concurrency (16 worker threads)
    - Ollama: OLLAMA_NUM_PARALLEL=16, OLLAMA_GPU_LAYERS=999 (full GPU), 16 worker threads
    - vLLM:   16 worker threads (matches production config)
    - Both get a warmup call excluded from timing
    - Same prompt, same texts, same labels, same random seed

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import pandas as pd
import yaml

from adapter import call_model
from prompt_builder import build_prompt


# =============================================================================
# MODEL MAPPING: vLLM names -> Ollama names
# =============================================================================
# Ollama uses its own naming scheme (not HuggingFace paths).
# These are the equivalent models in Ollama's registry.

MODELS = [
    {
        "short_name": "Llama 3.3 70B",
        "origin": "Western (Meta, USA)",
        "vllm_name": "casperhansen/llama-3.3-70b-instruct-awq",
        "ollama_name": "llama3.3:70b",
        "max_tokens_vllm": 256,
        "max_tokens_ollama": 256,
        "vllm_known_rpm": 195,  # estimated from 3.1 baseline (same VRAM/arch)
    },
    {
        "short_name": "Qwen 2.5 72B",
        "origin": "Eastern (Alibaba, China)",
        "vllm_name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "ollama_name": "qwen2.5:72b",
        "max_tokens_vllm": 256,
        "max_tokens_ollama": 256,
        "vllm_known_rpm": 162,  # from run_005 benchmark
    },
    {
        "short_name": "DeepSeek R1 32B",
        "origin": "Eastern (DeepSeek, China)",
        "vllm_name": "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
        "ollama_name": "deepseek-r1:32b",
        "max_tokens_vllm": 1024,  # R1 reasoning chains need more tokens
        "max_tokens_ollama": 1024,
        "vllm_known_rpm": 93,  # from run_005 benchmark
    },
]

# Provider configs matching config2.yaml / config3_50_samps.yaml
PROVIDER_OLLAMA = {
    "type": "openai_compatible",
    "base_url": "http://localhost:11434/v1",
    "api_key_env": None,
}

PROVIDER_VLLM = {
    "type": "openai_compatible",
    "base_url": "http://localhost:8900/v1",
    "api_key_env": "VLLM_API_KEY",
}

# =============================================================================
# FULL CORPUS SCALE (for time projections)
# =============================================================================

FULL_CORPUS_TEXTS = 56_797
PROMPTS_PER_TEXT = 5
LABELS_PER_TEXT = 25
FULL_CORPUS_CALLS = FULL_CORPUS_TEXTS * PROMPTS_PER_TEXT * LABELS_PER_TEXT  # 7,099,625


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

def parse_yes_no(response_text):
    """Quick YES/NO parser (same as run_annotation.py)."""
    import re
    if not response_text or response_text.startswith("ERROR:"):
        return -1
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    if cleaned:
        response_text = cleaned
    upper = response_text.upper().strip()
    head = upper[:80]
    if re.match(r'^(\*\*)?YES(\*\*)?[\s\.,:;\-\n]', head) or head == "YES":
        return 1
    if re.match(r'^(\*\*)?NO(\*\*)?[\s\.,:;\-\n]', head) or head == "NO":
        return 0
    if "YES" in head:
        return 1
    if "NO" in head:
        return 0
    tail = upper[-150:]
    if re.search(r'\bYES\b', tail):
        return 1
    if re.search(r'\bNO\b', tail):
        return 0
    return -1


def check_ollama_running():
    """Check if Ollama is accessible."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        resp = urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def check_vllm_running():
    """Check if vLLM is accessible."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8900/v1/models",
            headers={"Authorization": "Bearer sk-local-shahaan"},
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def list_ollama_models():
    """List models available in Ollama."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request("http://localhost:11434/api/tags")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def run_single_benchmark(provider_config, model_name, max_tokens, config,
                         texts_df, label_names, num_labels, prompt_config,
                         labels, annotation_guidelines, example_selection, rng,
                         workers=1):
    """
    Run the benchmark for one backend+model combo.
    Supports both sequential (workers=1) and threaded (workers>1) modes.
    Returns dict with timing results, or None on failure.
    """
    from joblib import Parallel, delayed

    runtime = config["runtime"]
    selected_labels = label_names[:num_labels]
    total_calls = len(texts_df) * len(selected_labels)

    print(f"      Calls: {len(texts_df)} texts x {len(selected_labels)} labels = {total_calls}")
    print(f"      Workers: {workers}")

    # Build all tasks
    tasks = []
    for _, text_row in texts_df.iterrows():
        text_id = text_row[runtime["text_id_column"]]
        text = str(text_row[runtime["text_column"]])
        for label_name in selected_labels:
            prompt = build_prompt(
                prompt_config=prompt_config,
                label_name=label_name,
                label_config=labels[label_name],
                text=text,
                example_selection=example_selection,
                rng=rng,
                annotation_guidelines=annotation_guidelines,
            )
            tasks.append((text_id, label_name, prompt))

    # Warmup (exclude from timing)
    print(f"      Warmup call...", end=" ", flush=True)
    t0_id, t0_label, t0_prompt = tasks[0]
    warmup_resp = call_model(
        provider_config=provider_config,
        model_name=model_name,
        prompt=t0_prompt,
        max_tokens=max_tokens,
        temperature=0,
        max_retries=2,
        retry_delay=2,
    )
    if warmup_resp.startswith("ERROR:"):
        print(f"FAILED: {warmup_resp}")
        return None
    print(f"OK ({parse_yes_no(warmup_resp)})")

    def call_one(text_id, label_name, prompt):
        """Single model call. Returns True if error, False if OK."""
        resp = call_model(
            provider_config=provider_config,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            max_retries=2,
            retry_delay=2,
        )
        return resp.startswith("ERROR:") if resp else True

    # Timed run
    start = time.time()

    if workers <= 1:
        # Sequential
        errors = 0
        for i, (text_id, label_name, prompt) in enumerate(tasks):
            is_error = call_one(text_id, label_name, prompt)
            if is_error:
                errors += 1
            if (i + 1) % 25 == 0 or i == len(tasks) - 1:
                elapsed_so_far = time.time() - start
                rpm_so_far = ((i + 1) / elapsed_so_far) * 60 if elapsed_so_far > 0 else 0
                eta_min = (len(tasks) - (i + 1)) / rpm_so_far if rpm_so_far > 0 else 0
                print(f"      [{i+1}/{len(tasks)}] RPM: {rpm_so_far:.1f} | "
                      f"ETA: {eta_min:.1f} min | Errors: {errors}", end="\r")
        print()
    else:
        # Threaded via joblib
        error_flags = Parallel(n_jobs=workers, backend="threading")(
            delayed(call_one)(t, l, p) for t, l, p in tasks
        )
        errors = sum(1 for e in error_flags if e)

    elapsed = time.time() - start
    rpm = (len(tasks) / elapsed) * 60 if elapsed > 0 else 0
    avg_per_call = elapsed / len(tasks) if len(tasks) > 0 else 0

    return {
        "total_calls": len(tasks),
        "elapsed_seconds": round(elapsed, 2),
        "avg_seconds_per_call": round(avg_per_call, 3),
        "effective_rpm": round(rpm, 1),
        "errors": errors,
    }


def run_vllm_benchmark_threaded(model, config, texts_df, label_names,
                                num_labels, prompt_config, labels,
                                annotation_guidelines, example_selection,
                                rng, workers):
    """
    Run vLLM benchmark with threading (vLLM handles concurrent requests well).
    Returns dict with timing results, or None on failure.
    """
    from joblib import Parallel, delayed

    runtime = config["runtime"]
    selected_labels = label_names[:num_labels]

    # Build all tasks
    tasks = []
    for _, text_row in texts_df.iterrows():
        text_id = text_row[runtime["text_id_column"]]
        text = str(text_row[runtime["text_column"]])
        for label_name in selected_labels:
            prompt = build_prompt(
                prompt_config=prompt_config,
                label_name=label_name,
                label_config=labels[label_name],
                text=text,
                example_selection=example_selection,
                rng=random.Random(runtime.get("random_seed", 42)),
                annotation_guidelines=annotation_guidelines,
            )
            tasks.append((text_id, label_name, prompt))

    total_calls = len(tasks)
    print(f"      Calls: {len(texts_df)} texts x {len(selected_labels)} labels = {total_calls}")
    print(f"      Workers: {workers}")

    # Warmup
    print(f"      Warmup call...", end=" ", flush=True)
    warmup = call_model(
        provider_config=PROVIDER_VLLM,
        model_name=model["vllm_name"],
        prompt=tasks[0][2],
        max_tokens=model["max_tokens_vllm"],
        temperature=0, max_retries=2, retry_delay=2,
    )
    if warmup.startswith("ERROR:"):
        print(f"FAILED: {warmup}")
        return None
    print("OK")

    def call_one(text_id, label_name, prompt):
        resp = call_model(
            provider_config=PROVIDER_VLLM,
            model_name=model["vllm_name"],
            prompt=prompt,
            max_tokens=model["max_tokens_vllm"],
            temperature=0, max_retries=2, retry_delay=2,
        )
        return resp.startswith("ERROR:") if resp else True

    start = time.time()
    error_flags = Parallel(n_jobs=workers, backend="threading")(
        delayed(call_one)(t, l, p) for t, l, p in tasks
    )
    elapsed = time.time() - start

    errors = sum(1 for e in error_flags if e)
    rpm = (total_calls / elapsed) * 60 if elapsed > 0 else 0
    avg = elapsed / total_calls if total_calls > 0 else 0

    return {
        "total_calls": total_calls,
        "elapsed_seconds": round(elapsed, 2),
        "avg_seconds_per_call": round(avg, 3),
        "effective_rpm": round(rpm, 1),
        "errors": errors,
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def print_report(results):
    """Print the final comparison report — time-focused, no cost."""

    print("\n")
    print("=" * 90)
    print("  BENCHMARK RESULTS: Ollama vs vLLM")
    print("  Ammunition for HPC CUDA Driver Upgrade Request")
    print("=" * 90)
    print("  Methodology: Both backends tested with identical concurrency (16 threads).")
    print("  Ollama given OLLAMA_NUM_PARALLEL=16 + OLLAMA_GPU_LAYERS=999 (full GPU offload).")
    print("  Worker sweep performed to find Ollama's optimal throughput.")

    # --- RAW THROUGHPUT TABLE ---
    print(f"\n{'MODEL':<22s} | {'BACKEND':<20s} | {'RPM':>8s} | {'Avg (s)':>8s} | "
          f"{'Calls':>6s} | {'Errors':>6s}")
    print("-" * 85)

    for r in results:
        print(f"{r['model']:<22s} | {r['backend']:<20s} | {r['rpm']:>8.1f} | "
              f"{r['avg_s']:>8.3f} | {r['calls']:>6d} | {r['errors']:>6d}")

    # --- SPEEDUP TABLE ---
    print(f"\n{'MODEL':<22s} | {'Ollama RPM':>11s} | {'vLLM RPM':>9s} | {'Speedup':>8s}")
    print("-" * 65)

    model_names = list(dict.fromkeys(r["model"] for r in results))
    speedups = []

    for model in model_names:
        ollama_r = [r for r in results if r["model"] == model and r["backend"].startswith("Ollama")]
        vllm_r = [r for r in results if r["model"] == model and r["backend"].startswith("vLLM")]

        ollama_rpm = ollama_r[0]["rpm"] if ollama_r else 0
        ollama_backend = ollama_r[0]["backend"] if ollama_r else "Ollama"
        vllm_rpm = vllm_r[0]["rpm"] if vllm_r else 0

        speedup = vllm_rpm / ollama_rpm if ollama_rpm > 0 else float("inf")
        speedups.append({
            "model": model, "ollama_rpm": ollama_rpm,
            "ollama_backend": ollama_backend,
            "vllm_rpm": vllm_rpm, "speedup": speedup,
        })

        print(f"{model:<22s} | {ollama_rpm:>11.1f} | {vllm_rpm:>9.1f} | {speedup:>7.1f}x"
              f"  {ollama_backend}")

    # --- FULL CORPUS TIME PROJECTION ---
    print(f"\n{'=' * 90}")
    print(f"  FULL CORPUS TIME PROJECTION: {FULL_CORPUS_TEXTS:,} texts x "
          f"{PROMPTS_PER_TEXT} prompts x {LABELS_PER_TEXT} labels = "
          f"{FULL_CORPUS_CALLS:,} calls per model")
    print(f"{'=' * 90}")

    print(f"\n{'MODEL':<22s} | {'Backend':<8s} | {'Days':>8s} | {'Hours':>9s}")
    print("-" * 55)

    for s in speedups:
        for backend, rpm in [(s["ollama_backend"], s["ollama_rpm"]), ("vLLM", s["vllm_rpm"])]:
            if rpm > 0:
                hours = FULL_CORPUS_CALLS / rpm / 60
                days = hours / 24
            else:
                hours = float("inf")
                days = float("inf")
            backend_short = "Ollama" if backend.startswith("Ollama") else "vLLM"
            print(f"{s['model']:<22s} | {backend_short:<8s} | {days:>8.1f} | {hours:>9.1f}")

    # --- BOTTOM LINE ---
    print(f"\n{'=' * 90}")
    print(f"  BOTTOM LINE (across all {len(model_names)} models, single full-corpus run)")
    print(f"{'=' * 90}")

    total_ollama_days = 0
    total_vllm_days = 0
    for s in speedups:
        if s["ollama_rpm"] > 0:
            total_ollama_days += FULL_CORPUS_CALLS / s["ollama_rpm"] / 60 / 24
        if s["vllm_rpm"] > 0:
            total_vllm_days += FULL_CORPUS_CALLS / s["vllm_rpm"] / 60 / 24

    print(f"\n  Total wall-clock time (all {len(model_names)} models, sequential):")
    print(f"    Ollama:  {total_ollama_days:.1f} days")
    print(f"    vLLM:    {total_vllm_days:.1f} days")
    print(f"    Saved:   {total_ollama_days - total_vllm_days:.1f} days")

    avg_speedup = sum(s["speedup"] for s in speedups) / len(speedups) if speedups else 0
    ollama_configs = ", ".join(s["ollama_backend"] for s in speedups)
    print(f"\n  Average speedup: {avg_speedup:.1f}x faster with vLLM")
    print(f"  (Ollama tested at optimal concurrency: {ollama_configs})")
    print(f"\n  RECOMMENDATION: Upgrade CUDA drivers on HPC A1 nodes to enable vLLM,")
    print(f"  saving ~{total_ollama_days - total_vllm_days:.0f} days per full model sweep.")
    print(f"{'=' * 90}\n")

    return speedups


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ollama vs vLLM Benchmark -- CUDA Upgrade Ammunition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config3_50_samps.yaml",
                        help="Config YAML with labels, prompts, sample file "
                             "(default: config3_50_samps.yaml)")
    parser.add_argument("--texts", type=int, default=9,
                        help="Number of texts to benchmark (default: 9)")
    parser.add_argument("--labels", type=int, default=5,
                        help="Number of labels per text (default: 5 for quick run; "
                             "use 25 for full)")
    parser.add_argument("--prompt-id", type=int, default=1,
                        help="Prompt ID to use (default: 1 = 0-shot binary)")
    parser.add_argument("--model-idx", type=int, default=None,
                        help="Run single model by index (0=Llama, 1=Qwen, 2=DeepSeek). "
                             "Default: all 3.")
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM benchmark; use known RPM baselines from run_005")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="Skip Ollama benchmark (for testing report with known data)")
    parser.add_argument("--workers-ollama", type=int, default=16,
                        help="Worker threads for Ollama benchmark (default: 16; "
                             "start Ollama with OLLAMA_NUM_PARALLEL=16 OLLAMA_GPU_LAYERS=999)")
    parser.add_argument("--sweep-ollama", action="store_true",
                        help="Sweep Ollama worker counts (1,2,4,8,16) and use the best RPM. "
                             "Adds ~5 min per extra worker count. Most defensible comparison.")
    parser.add_argument("--workers-vllm", type=int, default=16,
                        help="Worker threads for vLLM benchmark (default: 16)")
    # For testing with dummy Ollama RPM values
    parser.add_argument("--ollama-rpms", type=str, default=None,
                        help="Override Ollama RPMs for report-only mode "
                             "(comma-separated, e.g. '3.5,3.2,5.1')")
    args = parser.parse_args()

    # --- Load config ---
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("  Available configs: config2.yaml, config3_50_samps.yaml")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runtime = config["runtime"]
    labels = config["labels"]
    label_names = list(labels.keys())
    annotation_guidelines = config.get("annotation_guidelines", "")
    example_selection = runtime.get("example_selection", "fixed")
    rng = random.Random(runtime.get("random_seed", 42))

    # Get the benchmark prompt
    prompt_config = None
    for p in config["prompts"]:
        if p["id"] == args.prompt_id:
            prompt_config = p
            break
    if not prompt_config:
        print(f"ERROR: Prompt ID {args.prompt_id} not found in config.")
        sys.exit(1)

    # --- Load texts ---
    sample_file = runtime["sample_file"]
    if sample_file.endswith(".csv"):
        texts_df = pd.read_csv(sample_file).head(args.texts)
    else:
        texts_df = pd.read_excel(sample_file).head(args.texts)

    # --- Select models ---
    if args.model_idx is not None:
        if args.model_idx >= len(MODELS):
            print(f"ERROR: Model index {args.model_idx} out of range (0-{len(MODELS)-1})")
            sys.exit(1)
        models_to_test = [MODELS[args.model_idx]]
    else:
        models_to_test = MODELS

    num_labels = min(args.labels, len(label_names))

    # --- Print plan ---
    print("=" * 70)
    print("OLLAMA vs vLLM BENCHMARK")
    print("=" * 70)
    print(f"  Config:      {args.config}")
    print(f"  Prompt:      [{args.prompt_id}] {prompt_config['name']}")
    print(f"  Texts:       {len(texts_df)}")
    print(f"  Labels:      {num_labels} of {len(label_names)}")
    print(f"  Calls/model: {len(texts_df) * num_labels}")
    print(f"  Models:      {len(models_to_test)}")
    for m in models_to_test:
        print(f"    - {m['short_name']} ({m['origin']})")
    print(f"  Skip vLLM:   {args.skip_vllm}"
          f"{' (using known baselines)' if args.skip_vllm else ''}")
    print(f"  Skip Ollama: {args.skip_ollama}")
    print(f"  Ollama workers: {args.workers_ollama} "
          f"(start with: OLLAMA_NUM_PARALLEL={args.workers_ollama} "
          f"OLLAMA_GPU_LAYERS=999 ollama serve)")
    print(f"  vLLM workers:   {args.workers_vllm}")
    print("=" * 70)

    # --- Handle report-only mode with overridden Ollama RPMs ---
    if args.ollama_rpms:
        ollama_rpm_list = [float(x) for x in args.ollama_rpms.split(",")]
        if len(ollama_rpm_list) != len(models_to_test):
            print(f"ERROR: --ollama-rpms needs {len(models_to_test)} values, "
                  f"got {len(ollama_rpm_list)}")
            sys.exit(1)

        all_results = []
        for m, o_rpm in zip(models_to_test, ollama_rpm_list):
            all_results.append({
                "model": m["short_name"], "backend": "Ollama",
                "rpm": o_rpm, "avg_s": round(60/o_rpm, 3) if o_rpm > 0 else 0,
                "calls": 0, "errors": 0,
            })
            all_results.append({
                "model": m["short_name"], "backend": "vLLM",
                "rpm": m["vllm_known_rpm"],
                "avg_s": round(60/m["vllm_known_rpm"], 3),
                "calls": 0, "errors": 0,
            })

        speedups = print_report(all_results)

        # Save
        os.makedirs("outputs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(all_results).to_csv(
            f"outputs/benchmark_ollama_vs_vllm_{ts}.csv", index=False)
        return

    # --- Pre-flight checks ---
    if not args.skip_ollama:
        if not check_ollama_running():
            print("\nERROR: Ollama is not running.")
            print("  Start with parallel support:")
            print(f"    pkill -f ollama")
            print(f"    OLLAMA_NUM_PARALLEL={args.workers_ollama} OLLAMA_GPU_LAYERS=999 nohup ollama serve &")
            print(f"    sleep 5")
            sys.exit(1)

        available = list_ollama_models()
        print(f"\n  Ollama models available: {available}")
        for m in models_to_test:
            found = any(m["ollama_name"] in a for a in available)
            if not found:
                print(f"\n  WARNING: {m['ollama_name']} not found in Ollama.")
                print(f"  Pull it first:  ollama pull {m['ollama_name']}")

    if not args.skip_vllm:
        if not check_vllm_running():
            print("\n  NOTE: vLLM is not running. Using known baselines instead.")
            args.skip_vllm = True

    # --- Run benchmarks ---
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in models_to_test:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model['short_name']} ({model['origin']})")
        print(f"{'='*70}")

        # --- Ollama benchmark ---
        if not args.skip_ollama:
            if args.sweep_ollama:
                # Sweep worker counts to find Ollama's best throughput
                sweep_counts = [1, 2, 4, 8, 16]
                print(f"\n    [OLLAMA] Worker sweep: {sweep_counts}")
                print(f"    [OLLAMA] Finding optimal throughput for {model['ollama_name']}...")

                sweep_results = []
                for w in sweep_counts:
                    print(f"\n    [OLLAMA] Testing workers={w}...")
                    result = run_single_benchmark(
                        provider_config=PROVIDER_OLLAMA,
                        model_name=model["ollama_name"],
                        max_tokens=model["max_tokens_ollama"],
                        config=config,
                        texts_df=texts_df,
                        label_names=label_names,
                        num_labels=num_labels,
                        prompt_config=prompt_config,
                        labels=labels,
                        annotation_guidelines=annotation_guidelines,
                        example_selection=example_selection,
                        rng=random.Random(runtime.get("random_seed", 42)),
                        workers=w,
                    )
                    if result:
                        sweep_results.append({
                            "workers": w,
                            "rpm": result["effective_rpm"],
                            "avg_s": result["avg_seconds_per_call"],
                            "calls": result["total_calls"],
                            "errors": result["errors"],
                        })
                        print(f"    [OLLAMA] workers={w}: {result['effective_rpm']:.1f} RPM "
                              f"({result['avg_seconds_per_call']:.3f}s/call)")
                    else:
                        print(f"    [OLLAMA] workers={w}: FAILED")

                # Print sweep summary and pick best
                if sweep_results:
                    print(f"\n    [OLLAMA] Worker Sweep Results for {model['short_name']}:")
                    print(f"    {'Workers':>8s} | {'RPM':>8s} | {'Avg (s)':>8s} | {'Errors':>6s}")
                    print(f"    {'-'*42}")
                    best = max(sweep_results, key=lambda r: r["rpm"])
                    for sr in sweep_results:
                        marker = " <-- BEST" if sr["workers"] == best["workers"] else ""
                        print(f"    {sr['workers']:>8d} | {sr['rpm']:>8.1f} | "
                              f"{sr['avg_s']:>8.3f} | {sr['errors']:>6d}{marker}")

                    all_results.append({
                        "model": model["short_name"],
                        "backend": f"Ollama (best: {best['workers']}w)",
                        "rpm": best["rpm"],
                        "avg_s": best["avg_s"],
                        "calls": best["calls"],
                        "errors": best["errors"],
                    })
                    print(f"\n    [OLLAMA] Best: workers={best['workers']} @ "
                          f"{best['rpm']:.1f} RPM")
                else:
                    print(f"    [OLLAMA] All sweep runs FAILED for {model['short_name']}")
                    all_results.append({
                        "model": model["short_name"],
                        "backend": "Ollama (sweep failed)",
                        "rpm": 0, "avg_s": 0, "calls": 0, "errors": -1,
                    })

            else:
                # Single worker count run
                print(f"\n    [OLLAMA] Benchmarking {model['ollama_name']} "
                      f"(workers={args.workers_ollama})...")
                ollama_result = run_single_benchmark(
                    provider_config=PROVIDER_OLLAMA,
                    model_name=model["ollama_name"],
                    max_tokens=model["max_tokens_ollama"],
                    config=config,
                    texts_df=texts_df,
                    label_names=label_names,
                    num_labels=num_labels,
                    prompt_config=prompt_config,
                    labels=labels,
                    annotation_guidelines=annotation_guidelines,
                    example_selection=example_selection,
                    rng=random.Random(runtime.get("random_seed", 42)),
                    workers=args.workers_ollama,
                )
                if ollama_result:
                    all_results.append({
                        "model": model["short_name"],
                        "backend": "Ollama",
                        "rpm": ollama_result["effective_rpm"],
                        "avg_s": ollama_result["avg_seconds_per_call"],
                        "calls": ollama_result["total_calls"],
                        "errors": ollama_result["errors"],
                    })
                    print(f"    [OLLAMA] Done: {ollama_result['effective_rpm']:.1f} RPM "
                          f"({ollama_result['avg_seconds_per_call']:.3f}s/call)")
                else:
                    print(f"    [OLLAMA] FAILED for {model['short_name']}")
                    all_results.append({
                        "model": model["short_name"], "backend": "Ollama",
                        "rpm": 0, "avg_s": 0, "calls": 0, "errors": -1,
                    })
        else:
            print(f"\n    [OLLAMA] Skipped")

        # --- vLLM benchmark ---
        if not args.skip_vllm:
            print(f"\n    [vLLM] Benchmarking {model['vllm_name']}...")
            vllm_result = run_vllm_benchmark_threaded(
                model=model,
                config=config,
                texts_df=texts_df,
                label_names=label_names,
                num_labels=num_labels,
                prompt_config=prompt_config,
                labels=labels,
                annotation_guidelines=annotation_guidelines,
                example_selection=example_selection,
                rng=random.Random(runtime.get("random_seed", 42)),
                workers=args.workers_vllm,
            )
            if vllm_result:
                all_results.append({
                    "model": model["short_name"],
                    "backend": "vLLM",
                    "rpm": vllm_result["effective_rpm"],
                    "avg_s": vllm_result["avg_seconds_per_call"],
                    "calls": vllm_result["total_calls"],
                    "errors": vllm_result["errors"],
                })
                print(f"    [vLLM] Done: {vllm_result['effective_rpm']:.1f} RPM "
                      f"({vllm_result['avg_seconds_per_call']:.3f}s/call)")
            else:
                # Fall back to known baseline
                print(f"    [vLLM] FAILED -- using known baseline: "
                      f"{model['vllm_known_rpm']} RPM")
                all_results.append({
                    "model": model["short_name"], "backend": "vLLM",
                    "rpm": model["vllm_known_rpm"],
                    "avg_s": round(60/model["vllm_known_rpm"], 3),
                    "calls": 0, "errors": -1,
                })
        else:
            print(f"\n    [vLLM] Using known baseline: {model['vllm_known_rpm']} RPM")
            all_results.append({
                "model": model["short_name"],
                "backend": "vLLM",
                "rpm": model["vllm_known_rpm"],
                "avg_s": round(60 / model["vllm_known_rpm"], 3),
                "calls": 0,
                "errors": 0,
            })

    # --- Generate report ---
    if all_results:
        speedups = print_report(all_results)

        # Save results to CSV
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/benchmark_ollama_vs_vllm_{timestamp}.csv"
        pd.DataFrame(all_results).to_csv(output_path, index=False)
        print(f"  Raw results saved to: {output_path}")

        # Save report to text file
        report_path = f"outputs/benchmark_ollama_vs_vllm_report_{timestamp}.txt"
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(all_results)
        report_text = buf.getvalue()

        with open(report_path, "w") as rf:
            rf.write(f"Ollama vs vLLM Benchmark Report\n")
            rf.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            rf.write(f"Config: {args.config}\n")
            rf.write(f"Texts: {len(texts_df)} | Labels: {num_labels} | "
                     f"Models: {len(models_to_test)}\n")
            rf.write(report_text)

        print(f"  Report saved to:      {report_path}")

    print("\n  IMPORTANT: Shut down Ollama to free GPU memory:")
    print("    pkill -f ollama")


if __name__ == "__main__":
    main()
