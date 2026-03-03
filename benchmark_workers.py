#!/usr/bin/env python3
"""
Worker Sweep Benchmark
=======================
Tests different worker counts to find the optimal throughput for the
local vLLM server. Runs a small subset of texts and measures effective
RPM for each worker count.

Usage:
    python benchmark_workers.py                          # default sweep
    python benchmark_workers.py --texts 3                # use 3 texts
    python benchmark_workers.py --workers 1,2,4,8,16     # custom worker list
    python benchmark_workers.py --config config2.yaml    # alternate config
    python benchmark_workers.py --prompt-id 1            # single prompt only

Results are printed as a table and saved to outputs/worker_sweep.csv

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import os
import random
import time
from datetime import datetime

import pandas as pd
import yaml

from adapter import call_model
from prompt_builder import build_prompt


def parse_yes_no(response_text):
    """Quick YES/NO parser (same as run_annotation.py)."""
    import re
    if not response_text or response_text.startswith("ERROR:"):
        return -1
    # Strip <think>...</think> blocks (DeepSeek-R1 reasoning chains)
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


def run_benchmark(config, texts_df, num_workers, prompt_config, label_names,
                  labels, annotation_guidelines, example_selection, rng):
    """
    Run all texts × labels for a single prompt with the given worker count.
    Returns (total_calls, total_seconds, errors).
    """
    from joblib import Parallel, delayed

    runtime = config["runtime"]
    providers = config["providers"]
    model_config = [m for m in config["models"] if m.get("enabled", True)][0]
    provider_config = providers[model_config["provider"]]
    model_name = model_config["name"]
    max_tokens = model_config.get("max_tokens", 1024)
    temperature = model_config.get("temperature", 0)
    max_retries = runtime.get("max_retries", 3)
    retry_delay = runtime.get("retry_delay_seconds", 2)

    # Build all tasks: (text_id, label_name, prompt_string)
    tasks = []
    for _, text_row in texts_df.iterrows():
        text_id = text_row[runtime["text_id_column"]]
        text = str(text_row[runtime["text_column"]])

        for label_name in label_names:
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

    total_calls = len(tasks)

    def call_one(text_id, label_name, prompt):
        """Single model call. Returns (text_id, label_name, prediction, error)."""
        response = call_model(
            provider_config=provider_config,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        prediction = parse_yes_no(response)
        is_error = response.startswith("ERROR:") if response else True
        return text_id, label_name, prediction, is_error

    # --- Warmup call (exclude from timing) ---
    print(f"    Warmup call...", end=" ", flush=True)
    t0, l0, p0 = tasks[0]
    call_one(t0, l0, p0)
    print("done")

    # --- Timed run ---
    if num_workers == 1:
        # Sequential
        start = time.time()
        results = [call_one(t, l, p) for t, l, p in tasks]
        elapsed = time.time() - start
    else:
        # Threaded via joblib
        start = time.time()
        results = Parallel(n_jobs=num_workers, backend="threading")(
            delayed(call_one)(t, l, p) for t, l, p in tasks
        )
        elapsed = time.time() - start

    errors = sum(1 for _, _, _, e in results if e)
    return total_calls, elapsed, errors


def main():
    parser = argparse.ArgumentParser(description="Worker Sweep Benchmark")
    parser.add_argument("--config", default="config2.yaml",
                        help="Path to config YAML")
    parser.add_argument("--texts", type=int, default=2,
                        help="Number of texts to use (default: 2)")
    parser.add_argument("--workers", type=str, default="1,2,4,6,8,10,12,16",
                        help="Comma-separated worker counts to test")
    parser.add_argument("--prompt-id", type=int, default=1,
                        help="Single prompt ID to use for benchmark (default: 1)")
    args = parser.parse_args()

    worker_counts = [int(w) for w in args.workers.split(",")]

    # --- Load config ---
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
        return

    # Get the enabled model
    enabled_models = [m for m in config["models"] if m.get("enabled", True)]
    if not enabled_models:
        print("ERROR: No models enabled in config.")
        return
    model_name = enabled_models[0]["name"]

    # --- Load texts ---
    texts_df = pd.read_excel(runtime["sample_file"]).head(args.texts)
    total_calls_per_run = len(texts_df) * len(label_names)

    print("=" * 70)
    print("WORKER SWEEP BENCHMARK")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Prompt:      [{args.prompt_id}] {prompt_config['name']}")
    print(f"  Texts:       {len(texts_df)}")
    print(f"  Labels:      {len(label_names)}")
    print(f"  Calls/run:   {total_calls_per_run}")
    print(f"  Workers:     {worker_counts}")
    print("=" * 70)

    # --- Run sweep ---
    results = []

    for w in worker_counts:
        print(f"\n  Testing workers={w}...")

        calls, elapsed, errors = run_benchmark(
            config=config,
            texts_df=texts_df,
            num_workers=w,
            prompt_config=prompt_config,
            label_names=label_names,
            labels=labels,
            annotation_guidelines=annotation_guidelines,
            example_selection=example_selection,
            rng=rng,
        )

        rpm = (calls / elapsed) * 60 if elapsed > 0 else 0
        avg_per_call = elapsed / calls if calls > 0 else 0

        results.append({
            "workers": w,
            "total_calls": calls,
            "elapsed_seconds": round(elapsed, 2),
            "avg_per_call": round(avg_per_call, 3),
            "effective_rpm": round(rpm, 1),
            "errors": errors,
        })

        print(f"    {calls} calls in {elapsed:.1f}s | "
              f"avg {avg_per_call:.3f}s/call | "
              f"RPM: {rpm:.1f} | "
              f"errors: {errors}")

    # --- Results summary ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Workers':>8s} | {'Calls':>6s} | {'Time (s)':>9s} | "
          f"{'Avg (s)':>8s} | {'RPM':>8s} | {'Errors':>6s}")
    print("-" * 60)

    best = max(results, key=lambda r: r["effective_rpm"])

    for r in results:
        marker = " <-- BEST" if r["workers"] == best["workers"] else ""
        print(f"{r['workers']:>8d} | {r['total_calls']:>6d} | "
              f"{r['elapsed_seconds']:>9.2f} | {r['avg_per_call']:>8.3f} | "
              f"{r['effective_rpm']:>8.1f} | {r['errors']:>6d}{marker}")

    # --- Projection ---
    print("\n" + "=" * 70)
    print("PROJECTION FOR FULL DATASET (56,797 texts)")
    print("=" * 70)

    full_calls = 56797 * 5 * 25  # 5 prompts × 25 labels
    for r in results:
        if r["effective_rpm"] > 0:
            hours = full_calls / r["effective_rpm"] / 60
            days = hours / 24
            marker = " <-- BEST" if r["workers"] == best["workers"] else ""
            print(f"  workers={r['workers']:>2d}: {hours:>8.1f} hours "
                  f"({days:>5.1f} days){marker}")

    # --- Save results ---
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/worker_sweep_{timestamp}.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print(f"\n  RECOMMENDATION: Use --workers {best['workers']} "
          f"({best['effective_rpm']:.1f} RPM)")
    print("=" * 70)


if __name__ == "__main__":
    main()
