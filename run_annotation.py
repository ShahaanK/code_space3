#!/usr/bin/env python3
"""
CAMEL Corpus Annotation Pipeline - Runner
===========================================
Thin runner script that loads config.yaml, iterates over all enabled
models × prompts × texts × labels, calls the API, and writes results
to a wide-format CSV.

SYSTEM/USER SPLIT (v2 — March 2026):
  Now uses build_prompt_split() by default, which sends the static
  prompt prefix (role + guidelines + label definition) as a system
  message and only the annotated text as a user message. This enables
  vLLM's automatic prefix caching — all texts for the same prompt × label
  share cached KV states, giving major throughput gains at scale.

  Use --no-split to fall back to the legacy single-message mode.

Usage:
    python run_annotation.py                         # split mode (default)
    python run_annotation.py --no-split              # legacy single-message
    python run_annotation.py --concurrency 10        # 10 concurrent async calls
    python run_annotation.py --workers 16            # 16 joblib threads
    python run_annotation.py --dry-run               # log prompts, skip API
    python run_annotation.py --config my.yaml        # alternate config
    python run_annotation.py --subset 3              # only first N texts

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import asyncio
import os
import random
import re
import time
import glob
from datetime import datetime

import pandas as pd
import yaml

from adapter import call_model, call_model_async
from prompt_builder import build_prompt, build_prompt_split


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_yes_no(response_text):
    """
    Parse a YES/NO answer from a model response.

    Checks both the beginning and end of the response, since some prompts
    ask for analysis first with YES/NO at the end.

    Returns:
        1 for YES, 0 for NO, -1 for UNCLEAR/ERROR.
    """
    if not response_text or response_text.startswith("ERROR:"):
        return -1
    # Strip <think>...</think> blocks (DeepSeek-R1 reasoning chains)
    # Interesting Find
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    if cleaned:
        response_text = cleaned
    upper = response_text.upper().strip()

    # Check the first ~80 characters (for prompts that ask YES/NO first)
    head = upper[:80]

    if re.match(r'^(\*\*)?YES(\*\*)?[\s\.,:;\-\n]', head) or head == "YES":
        return 1
    if re.match(r'^(\*\*)?NO(\*\*)?[\s\.,:;\-\n]', head) or head == "NO":
        return 0

    if "YES" in head:
        return 1
    if "NO" in head:
        return 0

    # Check the last ~150 characters (for prompts that ask analysis first)
    tail = upper[-150:]

    # Look for definitive answer patterns at the end
    if re.search(r'(?:ANSWER[:\s]*|^|\n)(\*\*)?YES(\*\*)?[\s\.,:;\-]*$', tail):
        return 1
    if re.search(r'(?:ANSWER[:\s]*|^|\n)(\*\*)?NO(\*\*)?[\s\.,:;\-]*$', tail):
        return 0

    # Broader tail search
    if re.search(r'\bYES\b', tail):
        return 1
    if re.search(r'\bNO\b', tail):
        return 0

    return -1


# =============================================================================
# RESUMPTION HELPERS
# =============================================================================

def load_existing_results(output_path):
    """
    Load existing results CSV and return a set of completed row keys
    for skip_existing support.

    Key: (text_id, prompt_id, model)
    """
    completed = set()
    if os.path.exists(output_path):
        try:
            if output_path.endswith(".feather"):
                df = pd.read_feather(output_path)
            else:
                df = pd.read_csv(output_path)

            for _, row in df.iterrows():
                key = (row["text_id"], row["prompt_id"], row["model"])
                completed.add(key)
        except Exception:
            pass
    return completed


# =============================================================================
# LABEL ANNOTATION — SYNC, THREADED, AND ASYNC
# =============================================================================

def _annotate_single_label(label_name, labels, prompt_config, text,
                           provider_config, model_name, max_tokens,
                           temperature, max_retries, retry_delay,
                           example_selection, rng, annotation_guidelines,
                           use_split=True):
    """
    Annotate a single label for one text. Used by both sync and threaded modes.

    When use_split=True (default), sends the prompt as system + user messages
    for vLLM prefix caching. When False, sends as a single user message.

    Returns (label_name, prediction, response_text).
    """
    if use_split:
        system_prompt, user_prompt = build_prompt_split(
            prompt_config=prompt_config,
            label_name=label_name,
            label_config=labels[label_name],
            text=text,
            example_selection=example_selection,
            rng=rng,
            annotation_guidelines=annotation_guidelines,
        )
        # system_prompt is None if template couldn't be split
        response_text = call_model(
            provider_config=provider_config,
            model_name=model_name,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    else:
        prompt = build_prompt(
            prompt_config=prompt_config,
            label_name=label_name,
            label_config=labels[label_name],
            text=text,
            example_selection=example_selection,
            rng=rng,
            annotation_guidelines=annotation_guidelines,
        )
        response_text = call_model(
            provider_config=provider_config,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    return label_name, parse_yes_no(response_text), response_text


def annotate_labels_sync(label_names, labels, prompt_config, text,
                         provider_config, model_name, max_tokens, temperature,
                         max_retries, retry_delay, delay, example_selection,
                         rng, dry_run, annotation_guidelines="",
                         use_split=True):
    """
    Annotate all labels for one text sequentially.
    Returns dict of {label_name: (prediction, response_text)}.
    """
    results = {}
    for label_name in label_names:
        if dry_run:
            results[label_name] = (-1, "[DRY RUN]")
        else:
            ln, pred, resp = _annotate_single_label(
                label_name, labels, prompt_config, text,
                provider_config, model_name, max_tokens,
                temperature, max_retries, retry_delay,
                example_selection, rng, annotation_guidelines,
                use_split=use_split,
            )
            results[ln] = (pred, resp)
            if delay > 0:
                time.sleep(delay)

    return results


def annotate_labels_threaded(label_names, labels, prompt_config, text,
                             provider_config, model_name, max_tokens,
                             temperature, max_retries, retry_delay,
                             num_workers, example_selection, rng, dry_run,
                             annotation_guidelines="", use_split=True):
    """
    Annotate all labels for one text using joblib worker threads.
    Returns dict of {label_name: (prediction, response_text)}.
    """
    from joblib import Parallel, delayed

    if dry_run:
        return {ln: (-1, "[DRY RUN]") for ln in label_names}

    task_results = Parallel(n_jobs=num_workers, backend="threading")(
        delayed(_annotate_single_label)(
            label_name, labels, prompt_config, text,
            provider_config, model_name, max_tokens,
            temperature, max_retries, retry_delay,
            example_selection, rng, annotation_guidelines,
            use_split=use_split,
        )
        for label_name in label_names
    )

    results = {}
    for label_name, prediction, response_text in task_results:
        results[label_name] = (prediction, response_text)

    return results


async def annotate_labels_async(label_names, labels, prompt_config, text,
                                provider_config, model_name, max_tokens,
                                temperature, max_retries, retry_delay,
                                concurrency, example_selection, rng, dry_run,
                                annotation_guidelines="", use_split=True):
    """
    Annotate all labels for one text concurrently.
    Uses a semaphore to cap concurrent API calls.
    Returns dict of {label_name: (prediction, response_text)}.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def annotate_one(label_name):
        if dry_run:
            return label_name, -1, "[DRY RUN]"

        if use_split:
            system_prompt, user_prompt = build_prompt_split(
                prompt_config=prompt_config,
                label_name=label_name,
                label_config=labels[label_name],
                text=text,
                example_selection=example_selection,
                rng=rng,
                annotation_guidelines=annotation_guidelines,
            )
            async with semaphore:
                response_text = await call_model_async(
                    provider_config=provider_config,
                    model_name=model_name,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
        else:
            prompt = build_prompt(
                prompt_config=prompt_config,
                label_name=label_name,
                label_config=labels[label_name],
                text=text,
                example_selection=example_selection,
                rng=rng,
                annotation_guidelines=annotation_guidelines,
            )
            async with semaphore:
                response_text = await call_model_async(
                    provider_config=provider_config,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )

        return label_name, parse_yes_no(response_text), response_text

    tasks = [annotate_one(ln) for ln in label_names]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for result in task_results:
        if isinstance(result, Exception):
            results["_error"] = (-1, f"ERROR: {str(result)}")
        else:
            label_name, prediction, response_text = result
            results[label_name] = (prediction, response_text)

    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CAMEL Annotation Pipeline")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and log them without calling API")
    parser.add_argument("--subset", type=int, default=None,
                        help="Only process the first N texts")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Max concurrent async API calls per text "
                             "(for cloud providers like OpenRouter)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of joblib worker threads per text "
                             "(for local vLLM/Ollama, e.g. 16)")
    parser.add_argument("--no-split", action="store_true",
                        help="Disable system/user prompt split (legacy mode). "
                             "By default, prompts are split for vLLM prefix caching.")
    args = parser.parse_args()

    use_split = not args.no_split
    use_async = args.concurrency > 1
    use_threads = args.workers > 1

    if use_async and use_threads:
        print("ERROR: Use either --concurrency (async) or --workers (threads), not both.")
        return

    # --- Load config ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    runtime = config["runtime"]
    providers = config["providers"]
    labels = config["labels"]
    label_names = list(labels.keys())

    # Filter enabled models and prompts
    enabled_models = [m for m in config["models"] if m.get("enabled", True)]
    enabled_prompts = [p for p in config["prompts"] if p.get("enabled", True)]

    if not enabled_models:
        print("ERROR: No models are enabled in config.")
        return
    if not enabled_prompts:
        print("ERROR: No prompts are enabled in config.")
        return

    # --- Setup RNG ---
    example_selection = runtime.get("example_selection", "fixed")
    rng = random.Random(runtime.get("random_seed", 42))

    # --- Load annotation guidelines (optional, used by config2) ---
    annotation_guidelines = config.get("annotation_guidelines", "")

    # --- Load texts ---
    sample_file = runtime["sample_file"]
    text_id_col = runtime["text_id_column"]
    text_col = runtime["text_column"]

    print(f"Loading texts from: {sample_file}")
    texts_df = pd.read_excel(sample_file) if sample_file.endswith((".xlsx", ".xls")) else pd.read_csv(sample_file)

    if args.subset:
        texts_df = texts_df.head(args.subset)
        print(f"  Subset mode: processing first {args.subset} texts")

    print(f"  Loaded {len(texts_df)} texts")

    # --- Prepare output ---
    output_dir = runtime.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = runtime.get("output_prefix", "camel_results")
    output_path = os.path.join(output_dir, f"{output_prefix}_{timestamp}.feather")
    skip_existing = runtime.get("skip_existing", True)
    completed = set()

    # Load completed work from any existing CSVs in the output directory
    if skip_existing:
        existing_files = sorted(
            glob.glob(os.path.join(output_dir, f"{output_prefix}_*.feather"))
            + glob.glob(os.path.join(output_dir, f"{output_prefix}_*.csv"))
        )
        for fpath in existing_files:
            loaded = load_existing_results(fpath)
            completed.update(loaded)

        if completed:
            print(f"  Checkpoint: found {len(completed)} completed rows, will skip")

    # --- Build column structure ---
    meta_cols = ["text_id", "prompt_id", "prompt_name", "model", "provider",
                 "temperature", "run_number"]
    label_cols = label_names
    response_cols = [f"response__{ln}" for ln in label_names]
    all_cols = meta_cols + label_cols + response_cols

    results_buffer = []
    all_results = []

    # --- Rate limiting ---
    delay = runtime.get("delay_between_calls_seconds", 0.5)
    max_retries = runtime.get("max_retries", 3)
    retry_delay = runtime.get("retry_delay_seconds", 2)

    # --- Compute totals ---
    total_rows = len(texts_df) * len(enabled_models) * len(enabled_prompts)
    total_api_calls = total_rows * len(label_names)

    if use_async:
        mode_str = f"{args.concurrency} (async)"
    elif use_threads:
        mode_str = f"{args.workers} (threaded/joblib)"
    else:
        mode_str = "1 (sequential)"

    print(f"\nPipeline configuration:")
    print(f"  Models:      {len(enabled_models)} enabled")
    print(f"  Prompts:     {len(enabled_prompts)} enabled")
    print(f"  Labels:      {len(label_names)}")
    print(f"  Texts:       {len(texts_df)}")
    print(f"  Parallelism: {mode_str}")
    print(f"  Prompt split: {'ON (system/user for prefix caching)' if use_split else 'OFF (legacy single message)'}")
    print(f"  Total output rows: {total_rows}")
    print(f"  Total API calls:   {total_api_calls}")

    if args.dry_run:
        print(f"\n  *** DRY RUN MODE — no API calls will be made ***\n")
    else:
        print(f"\n  Output: {output_path}\n")

    print("=" * 70)

    # --- Main loop ---
    row_count = 0
    call_count = 0
    start_time = time.time()

    # Prepare the async event loop if needed
    loop = asyncio.new_event_loop() if use_async else None

    for model_config in enabled_models:
        model_name = model_config["name"]
        provider_name = model_config["provider"]
        provider_config = providers[provider_name]
        temperature = model_config.get("temperature", 0.5)
        max_tokens = model_config.get("max_tokens", 300)

        print(f"\nModel: {model_name} (via {provider_name})")

        for prompt_config in enabled_prompts:
            prompt_id = prompt_config["id"]
            prompt_name = prompt_config["name"]

            print(f"  Prompt [{prompt_id}]: {prompt_name}")

            for _, text_row in texts_df.iterrows():
                text_id = text_row[text_id_col]
                text = str(text_row[text_col])

                # Skip if already completed
                row_key = (text_id, prompt_id, model_name)
                if skip_existing and row_key in completed:
                    continue

                row_count += 1
                elapsed = time.time() - start_time
                rate = row_count / elapsed if elapsed > 0 else 0
                remaining = (total_rows - row_count) / rate if rate > 0 else 0
                print(f"    Text {text_id} ({row_count}/{total_rows})"
                      f"  [{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

                # Build the row dict with metadata
                row = {
                    "text_id": text_id,
                    "prompt_id": prompt_id,
                    "prompt_name": prompt_name,
                    "model": model_name,
                    "provider": provider_name,
                    "temperature": temperature,
                    "run_number": 1,
                }

                # --- Annotate all labels ---
                if use_async:
                    label_results = loop.run_until_complete(
                        annotate_labels_async(
                            label_names=label_names,
                            labels=labels,
                            prompt_config=prompt_config,
                            text=text,
                            provider_config=provider_config,
                            model_name=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            max_retries=max_retries,
                            retry_delay=retry_delay,
                            concurrency=args.concurrency,
                            example_selection=example_selection,
                            rng=rng,
                            dry_run=args.dry_run,
                            annotation_guidelines=annotation_guidelines,
                            use_split=use_split,
                        )
                    )
                elif use_threads:
                    label_results = annotate_labels_threaded(
                        label_names=label_names,
                        labels=labels,
                        prompt_config=prompt_config,
                        text=text,
                        provider_config=provider_config,
                        model_name=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        num_workers=args.workers,
                        example_selection=example_selection,
                        rng=rng,
                        dry_run=args.dry_run,
                        annotation_guidelines=annotation_guidelines,
                        use_split=use_split,
                    )
                else:
                    label_results = annotate_labels_sync(
                        label_names=label_names,
                        labels=labels,
                        prompt_config=prompt_config,
                        text=text,
                        provider_config=provider_config,
                        model_name=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        delay=delay,
                        example_selection=example_selection,
                        rng=rng,
                        dry_run=args.dry_run,
                        annotation_guidelines=annotation_guidelines,
                        use_split=use_split,
                    )

                # --- Fill row from results ---
                for label_name in label_names:
                    if label_name in label_results:
                        prediction, response_text = label_results[label_name]
                    else:
                        prediction, response_text = -1, "ERROR: missing result"

                    row[label_name] = prediction
                    row[f"response__{label_name}"] = (
                        response_text[:500] if response_text else ""
                    )

                    if not args.dry_run:
                        call_count += 1

                # Mark completed
                completed.add(row_key)
                results_buffer.append(row)
                all_results.append(row)

                if len(all_results) % 50 == 0:
                    _flush_buffer(all_results, all_cols, output_path)
                    results_buffer = []

    # Clean up async loop
    if loop:
        loop.close()

    # Final flush
    if all_results:
        _flush_buffer(all_results, all_cols, output_path)
    # --- Summary ---
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Rows written:    {row_count}")
    print(f"  API calls made:  {call_count}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/60:.1f}m)")
    if call_count > 0:
        print(f"  Avg per call:    {total_time/call_count:.2f}s")
        print(f"  Effective RPM:   {call_count/(total_time/60):.0f}")
    if not args.dry_run:
        print(f"  Output: {output_path}")
    print(f"  Prompt split:    {'ON' if use_split else 'OFF'}")
    print("=" * 70)

    # --- Dry run: show a sample prompt (split view) ---
    if args.dry_run and enabled_prompts and len(texts_df) > 0:
        sample_prompt_config = enabled_prompts[0]
        sample_label = label_names[0]
        sample_text = str(texts_df.iloc[0][text_col])
        if len(sample_text) > 200:
            sample_text = sample_text[:200] + "..."

        if use_split:
            sys_p, usr_p = build_prompt_split(
                prompt_config=sample_prompt_config,
                label_name=sample_label,
                label_config=labels[sample_label],
                text=sample_text,
                example_selection=example_selection,
                rng=rng,
                annotation_guidelines=annotation_guidelines,
            )
            print(f"\n{'='*70}")
            print(f"SAMPLE SPLIT PROMPT (Prompt {sample_prompt_config['id']}, "
                  f"Label: {sample_label})")
            print(f"{'='*70}")
            print(f"\n--- SYSTEM MESSAGE (cached per prompt×label) ---")
            print(sys_p if sys_p else "[NOT SPLIT — using legacy mode]")
            print(f"\n--- USER MESSAGE (varies per text) ---")
            print(usr_p)
            print(f"{'='*70}")
        else:
            sample_prompt = build_prompt(
                prompt_config=sample_prompt_config,
                label_name=sample_label,
                label_config=labels[sample_label],
                text=sample_text,
                example_selection=example_selection,
                rng=rng,
                annotation_guidelines=annotation_guidelines,
            )
            print(f"\n{'='*70}")
            print(f"SAMPLE PROMPT (Prompt {sample_prompt_config['id']}, "
                  f"Label: {sample_label})")
            print(f"{'='*70}")
            print(sample_prompt)
            print(f"{'='*70}")


def _flush_buffer(buffer, columns, output_path):
    """Write all accumulated results to Feather (full overwrite each time)."""
    df = pd.DataFrame(buffer, columns=columns)
    df.reset_index(drop=True).to_feather(output_path)

if __name__ == "__main__":
    main()
