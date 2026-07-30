#!/usr/bin/env python3
"""
CAMEL Annotation — HPC Self-Contained Script
==============================================
Adapted from Prof. Introne's belief_extract_hpc.py for CAMEL annotation.

This script is designed to run as a single HTCondor job:
  1. Starts a vLLM server (via Singularity container)
  2. Waits for it to be ready
  3. Loads a data chunk (Feather format)
  4. Annotates each text × prompt × label using the CAMEL framework
  5. Saves results as Feather
  6. Shuts down vLLM

All prompt templates, label definitions, and annotation guidelines are
loaded from config2.yaml (shipped alongside this script).

Uses system/user prompt split for vLLM prefix caching — all texts for
the same prompt × label share cached KV states.

Usage:
    python camel_annotate_hpc.py chunks/chunk_000.feather results/results_000.feather

    # Custom model (default: Llama 3.3 70B)
    python camel_annotate_hpc.py chunk.feather result.feather --model Qwen/Qwen2.5-72B-Instruct-AWQ

    # Custom config
    python camel_annotate_hpc.py chunk.feather result.feather --config my_config.yaml

    # Test mode (first N texts only)
    python camel_annotate_hpc.py chunk.feather result.feather --test 10

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_MODEL = "casperhansen/llama-3.3-70b-instruct-awq"
DEFAULT_CONFIG = "config2.yaml"
DEFAULT_QUANTIZATION = "awq_marlin"
DEFAULT_TP = 2
DEFAULT_MAX_MODEL_LEN = 4096   # bumped 2048->4096: id-4 (multi-shot_binary_reasoning)
                               # all-examples prompts overflow 2048 on long texts +
                               # high-example labels (measured 2441 tok worst case)
DEFAULT_GPU_UTIL = 0.90
DEFAULT_NUM_THREADS = 16
DEFAULT_MAX_TOKENS = 256         # Llama/Qwen; override for DeepSeek
DEFAULT_VLLM_PORT = 8900
DEFAULT_CONTAINER = "vllm-openai.sif"

# DeepSeek needs more output tokens for reasoning chains
DEEPSEEK_MAX_TOKENS = 2048
DEEPSEEK_MAX_MODEL_LEN = 4096

# DeepSeek-R1 collapses into repetition loops under greedy decoding (temp 0),
# and a repetition penalty (1.15) traded that for token-level incoherence.
# Final decision: DeepSeek runs vendor sampling (temp 0.6) as a DOCUMENTED
# EXCEPTION to the temp-0 protocol; no repetition penalty. Constant retained
# for provenance but no longer applied (see MODEL_OVERRIDES / WORKLOG).
DEEPSEEK_REPETITION_PENALTY = 1.15

# Model-specific overrides
MODEL_OVERRIDES = {
    "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ": {
        "max_tokens": DEEPSEEK_MAX_TOKENS,
        "max_model_len": DEEPSEEK_MAX_MODEL_LEN,
        "temperature": 0.6,
        "top_p": 0.95,
        "min_p": 0.05,
        "extra_flags": "--trust-remote-code",
    },
    "Qwen/Qwen2.5-72B-Instruct-AWQ": {
        "extra_flags": "--trust-remote-code",
    },
    # HF repo ids are case-sensitive: the published repo is "-Chat", not "-chat".
    # No AWQ/GPTQ/INT4 release exists for this model, so it runs unquantized
    # (fp16) at TP=2 across both A100s on a node; see batch_acegpt-70b.sub.
    "FreedomIntelligence/AceGPT-v2-70B-Chat": {
        "extra_flags": "--trust-remote-code",
    },
    # Official TII GPTQ-Int4 release (model_type falcon_h1, quant_method gptq,
    # bits 4). Keyed separately from the unquantized repo because MODEL_OVERRIDES
    # matches the exact CAMEL_MODEL string.
    "tiiuae/Falcon-H1-34B-Instruct-GPTQ-Int4": {
        "extra_flags": "--trust-remote-code",
    },
    "tiiuae/Falcon-H1-34B-Instruct": {
        "extra_flags": "--trust-remote-code",
    },
}


# =============================================================================
# PROMPT BUILDING (self-contained, matches prompt_builder.py)
# =============================================================================

def build_prompt_split(template, label_name, label_definition,
                       markers_section, examples_section,
                       annotation_guidelines, text):
    """
    Build (system_prompt, user_prompt) for prefix caching.

    Everything before === TEXT === goes into system (cached).
    Everything from === TEXT === onward goes into user (per-text).
    """
    text_marker = "=== TEXT ==="

    if text_marker not in template:
        # Can't split — return as single user message
        prompt = template.format(
            text=text, label_name=label_name,
            label_definition=label_definition,
            markers_section=markers_section,
            examples_section=examples_section,
            annotation_guidelines=annotation_guidelines,
        )
        return None, _clean(prompt)

    idx = template.index(text_marker)
    sys_template = template[:idx].rstrip()
    usr_template = template[idx:]

    sys_prompt = sys_template.format(
        label_name=label_name, label_definition=label_definition,
        markers_section=markers_section, examples_section=examples_section,
        annotation_guidelines=annotation_guidelines, text="",
    )
    usr_prompt = usr_template.format(
        text=text, label_name=label_name, label_definition=label_definition,
        markers_section=markers_section, examples_section=examples_section,
        annotation_guidelines=annotation_guidelines,
    )

    return _clean(sys_prompt), _clean(usr_prompt)


def build_sections(prompt_config, label_config, label_name,
                   example_selection="fixed"):
    """Build markers and examples sections from config."""
    markers_section = ""
    if prompt_config.get("include_markers", False):
        markers = label_config.get("markers", "").strip()
        if markers:
            markers_section = f"Key markers: {markers}"

    examples_section = ""
    if prompt_config.get("include_examples", False):
        examples = label_config.get("examples", [])
        num = prompt_config.get("num_examples", 0)
        # example_selection == "all" injects every example for the label,
        # mirroring prompt_builder.select_examples on the Apophis path;
        # num_examples is then ignored. Otherwise cap to the first num
        # (num <= 0 falls through to "no cap => all").
        if example_selection != "all" and num > 0:
            examples = examples[:num]
        if examples:
            lines = [f"Example of {label_name}:"]
            for ex in examples:
                lines.append(f"  - {ex}")
            examples_section = "\n".join(lines)

    return markers_section, examples_section


def _clean(text):
    """Collapse multiple blank lines, strip."""
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# =============================================================================
# RESPONSE PARSING (matches run_annotation.py)
# =============================================================================

def parse_yes_no(response_text):
    """Parse YES/NO from model response. Returns 1, 0, or -1."""
    if not response_text or response_text.startswith("ERROR:"):
        return -1

    # Strip DeepSeek <think> blocks
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


# =============================================================================
# vLLM SERVER MANAGEMENT
# =============================================================================

def start_vllm(model, container_path, port, quantization, tp,
               gpu_util, max_model_len, extra_flags="",
               log_path="vllm_server.log"):
    """Start vLLM server via Singularity container. Returns subprocess."""
    cmd = [
        "singularity", "exec", "--nv", container_path,
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
    ]
    # Unquantized (fp16/bf16) checkpoints must OMIT --quantization entirely so
    # vLLM auto-detects from the model config. Passing a quant method the
    # weights do not carry is a hard startup failure. AceGPT-v2-70B-Chat is the
    # only roster model with no INT4 release, hence CAMEL_QUANTIZATION=none.
    if quantization and quantization.lower() not in ("none", "auto", ""):
        cmd += ["--quantization", quantization]
    cmd += [
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_model_len),
        # "--kv-cache-dtype", "fp8_e4m3",  # V1 experiment: removed to allow V1 engine
        "--port", str(port),
    ]
    if extra_flags:
        cmd.extend(extra_flags.split())

    print(f"  Starting vLLM: {model}")
    print(f"  Container: {container_path}")
    print(f"  Quantization: {quantization or 'auto (unquantized)'}, "
          f"TP: {tp}, GPU util: {gpu_util}")
    print(f"  Max model len: {max_model_len}, Port: {port}")

    # Per-job path: every job's cwd is the shared initialdir (hpc/), so a fixed
    # filename means all concurrent jobs open the SAME file with mode "w" and
    # interleave into it — destroying the vLLM diagnostics (startup failures,
    # the V0/V1 engine line) exactly when a 100+ job wave needs them most.
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_vllm(port, timeout=600, poll_interval=10):
    """Wait for vLLM to be ready."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/v1/models"
    start = time.time()

    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            if data.get("data"):
                elapsed = time.time() - start
                model_id = data["data"][0]["id"]
                print(f"  vLLM ready in {elapsed:.0f}s (model: {model_id})")
                return True
        except Exception:
            pass
        time.sleep(poll_interval)

    print(f"  ERROR: vLLM did not start within {timeout}s")
    return False


def kill_vllm(proc):
    """Kill vLLM process group."""
    if proc and proc.poll() is None:
        print("  Shutting down vLLM...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
            except Exception:
                pass


# =============================================================================
# API CALL
# =============================================================================

def call_vllm(model, port, system_prompt, user_prompt, max_tokens,
              temperature=0, repetition_penalty=None,
              top_p=None, min_p=None, max_retries=3):
    """Call vLLM's OpenAI-compatible API. Returns response text."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/v1/chat/completions"

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]

    payload_dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Sampling params only sent when a model override sets them (e.g.
    # DeepSeek-R1 vendor sampling). Models without them see no change:
    # temperature defaults to 0; top_p/min_p/repetition_penalty are omitted.
    if repetition_penalty is not None:
        payload_dict["repetition_penalty"] = repetition_penalty
    if top_p is not None:
        payload_dict["top_p"] = top_p
    if min_p is not None:
        payload_dict["min_p"] = min_p

    payload = json.dumps(payload_dict).encode("utf-8")

    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                return f"ERROR: {str(e)}"


# =============================================================================
# CHECKPOINTING (crash-resumable shard I/O)
# =============================================================================

def _atomic_write_feather(df, final_path):
    """Write df to a temp file, fsync, then atomically os.replace into place.

    A mid-write crash can only ever leave a *.tmp file behind, never a
    half-written shard/output that a resume would trust. fsync of the file
    and the containing directory makes both the bytes and the rename durable
    across a node reschedule on the shared filesystem.
    """
    final_path = str(final_path)
    tmp_path = final_path + ".tmp"
    df.reset_index(drop=True).to_feather(tmp_path)
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, final_path)
    dir_fd = os.open(os.path.dirname(final_path) or ".", os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def shard_glob(results_dir, chunk_stem):
    """Sorted list of shard paths for one chunk: {chunk_stem}.part_*.feather."""
    import glob
    pattern = os.path.join(results_dir, f"{chunk_stem}.part_*.feather")
    return sorted(glob.glob(pattern))


def scan_completed_shards(results_dir, chunk_stem):
    """Scan existing shards for this chunk to support crash resume.

    Returns (completed, next_index, shard_paths):
      completed   : set of (int text_id, int prompt_id) already durably written
      next_index  : next shard index to write (max existing index + 1)
      shard_paths : the shard files found (sorted)
    """
    shard_paths = shard_glob(results_dir, chunk_stem)
    completed = set()
    next_index = 0
    for p in shard_paths:
        # Derive next index from the filename regardless of readability, so a
        # torn shard never causes a numbering collision.
        base = os.path.basename(p)
        try:
            idx = int(base.rsplit(".part_", 1)[1].split(".feather")[0])
            next_index = max(next_index, idx + 1)
        except (IndexError, ValueError):
            pass
        try:
            df = pd.read_feather(p)
        except Exception as e:
            # Atomic writes make this near-impossible, but if a shard is
            # unreadable, don't trust it — let its rows re-run.
            print(f"  WARNING: could not read shard {p}: {e}")
            continue
        for tid, pid in zip(df["text_id"], df["prompt_id"]):
            completed.add((int(tid), int(pid)))
    return completed, next_index, shard_paths


# =============================================================================
# ANNOTATION ENGINE
# =============================================================================

def annotate_chunk(chunk_df, config, model, port, max_tokens, num_threads,
                   text_id_col, text_col, repetition_penalty=None,
                   temperature=0, top_p=None, min_p=None,
                   results_dir=".", chunk_stem="results",
                   ckpt_every=25, completed=None, start_index=0):
    """
    Annotate all texts in a chunk for all enabled prompts × all labels.

    Returns a list of row dicts (one per text × prompt) in wide format,
    matching run_annotation.py output structure.
    """
    labels = config["labels"]
    label_names = list(labels.keys())
    annotation_guidelines = config.get("annotation_guidelines", "")
    # Honor runtime.example_selection ("all" => inject every example per label),
    # matching run_annotation.py on the Apophis path. camel_annotate_hpc.py
    # historically ignored this key, so num_examples silently meant different
    # things per code path; threading it here makes both runners identical.
    example_selection = config.get("runtime", {}).get("example_selection", "fixed")
    enabled_prompts = [p for p in config["prompts"] if p.get("enabled", True)]

    total_texts = len(chunk_df)
    total_prompts = len(enabled_prompts)
    total_calls = total_texts * total_prompts * len(label_names)

    print(f"\n  Annotation plan:")
    print(f"    Texts:   {total_texts}")
    print(f"    Prompts: {total_prompts}")
    print(f"    Labels:  {len(label_names)}")
    print(f"    Calls:   {total_calls}")
    print(f"    Threads: {num_threads}")

    completed = completed or set()
    buffer = []
    shard_index = start_index
    texts_in_buffer = 0
    new_rows = 0
    skipped_rows = 0

    def flush_buffer():
        """Durably write the current buffer as the next shard (atomic)."""
        nonlocal buffer, shard_index, texts_in_buffer
        if not buffer:
            return
        shard_path = os.path.join(
            results_dir, f"{chunk_stem}.part_{shard_index:03d}.feather")
        _atomic_write_feather(pd.DataFrame(buffer), shard_path)
        print(f"\n    checkpoint: wrote {len(buffer)} rows "
              f"-> {os.path.basename(shard_path)}")
        buffer = []
        texts_in_buffer = 0
        shard_index += 1

    global_start = time.time()
    call_count = 0

    for text_idx, (_, text_row) in enumerate(chunk_df.iterrows()):
        text_id = text_row[text_id_col]
        text = str(text_row[text_col])
        text_had_new = False

        for prompt_config in enabled_prompts:
            prompt_id = prompt_config["id"]
            prompt_name = prompt_config["name"]

            # Resume: skip rows already durably written by a prior (crashed) run
            if (int(text_id), int(prompt_id)) in completed:
                skipped_rows += 1
                continue

            # Build row metadata
            row = {
                "text_id": text_id,
                "prompt_id": prompt_id,
                "prompt_name": prompt_name,
                "model": model,
                "provider": "hpc_vllm",
                "temperature": temperature,
                "run_number": 1,
            }

            # Preserve original_index if present (for merge_results.py)
            if "original_index" in text_row.index:
                row["original_index"] = text_row["original_index"]

            # --- Annotate all labels in parallel ---
            def annotate_label(label_name):
                markers_section, examples_section = build_sections(
                    prompt_config, labels[label_name], label_name,
                    example_selection)

                sys_prompt, usr_prompt = build_prompt_split(
                    template=prompt_config["template"],
                    label_name=label_name,
                    label_definition=labels[label_name].get("definition", "").strip(),
                    markers_section=markers_section,
                    examples_section=examples_section,
                    annotation_guidelines=annotation_guidelines,
                    text=text,
                )

                response = call_vllm(model, port, sys_prompt, usr_prompt,
                                     max_tokens,
                                     temperature=temperature,
                                     repetition_penalty=repetition_penalty,
                                     top_p=top_p, min_p=min_p)
                prediction = parse_yes_no(response)
                return label_name, prediction, response

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(annotate_label, ln): ln
                    for ln in label_names
                }
                for future in as_completed(futures):
                    label_name, prediction, response = future.result()
                    row[label_name] = prediction
                    row[f"response__{label_name}"] = (
                        response[:4000] if response else "")
                    call_count += 1

            buffer.append(row)
            new_rows += 1
            text_had_new = True

        # Progress
        elapsed = time.time() - global_start
        rpm = (call_count / elapsed) * 60 if elapsed > 0 else 0
        pct = (text_idx + 1) / total_texts * 100
        print(f"    [{text_idx+1}/{total_texts}] ({pct:.0f}%) "
              f"RPM: {rpm:.0f} | "
              f"Elapsed: {elapsed:.0f}s", end="\r")

        # Durable checkpoint every ckpt_every newly-annotated texts
        if text_had_new:
            texts_in_buffer += 1
            if texts_in_buffer >= ckpt_every:
                flush_buffer()

    flush_buffer()  # tail: flush remaining rows as the final shard

    elapsed = time.time() - global_start
    rpm = (call_count / elapsed) * 60 if elapsed > 0 else 0
    print(f"\n\n  Annotation complete:")
    print(f"    Total calls:       {call_count}")
    print(f"    New rows:          {new_rows}")
    print(f"    Skipped (resumed): {skipped_rows}")
    print(f"    Wall time:         {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"    Effective RPM:     {rpm:.0f}")

    return new_rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CAMEL Annotation — HPC Self-Contained Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input chunk Feather file")
    parser.add_argument("output", help="Output results Feather file")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Config YAML with prompts/labels (default: {DEFAULT_CONFIG})")
    parser.add_argument("--container", default=DEFAULT_CONTAINER,
                        help=f"Singularity container (default: {DEFAULT_CONTAINER})")
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION,
                        help=f"Quantization method (default: {DEFAULT_QUANTIZATION})")
    parser.add_argument("--tp", type=int, default=DEFAULT_TP,
                        help=f"Tensor parallel size (default: {DEFAULT_TP})")
    parser.add_argument("--threads", type=int, default=DEFAULT_NUM_THREADS,
                        help=f"Worker threads for annotation (default: {DEFAULT_NUM_THREADS})")
    parser.add_argument("--port", type=int, default=DEFAULT_VLLM_PORT,
                        help=f"vLLM server port (default: {DEFAULT_VLLM_PORT})")
    parser.add_argument("--test", type=int, default=None,
                        help="Only process first N texts (for testing)")
    parser.add_argument("--text-id-col", default=None,
                        help="Text ID column name (auto-detected from config)")
    parser.add_argument("--text-col", default=None,
                        help="Text column name (auto-detected from config)")
    parser.add_argument("--no-server", action="store_true",
                        help="Skip starting vLLM (use already-running server). "
                             "For testing on Apophis without Singularity.")
    args = parser.parse_args()

    print("=" * 70)
    print("CAMEL Annotation — HPC Job")
    print("=" * 70)

    # --- Load config ---
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Column names from config or args
    text_id_col = args.text_id_col or config["runtime"].get("text_id_column", "Number")
    text_col = args.text_col or config["runtime"].get("text_column", "text")

    # --- Load input chunk ---
    print(f"\n  Input:  {args.input}")
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    chunk_df = pd.read_feather(args.input)
    print(f"  Rows:   {len(chunk_df)}")
    print(f"  Columns: {list(chunk_df.columns)[:8]}...")

    if args.test:
        chunk_df = chunk_df.head(args.test)
        print(f"  Test mode: processing first {args.test} texts")

    # --- Model-specific overrides ---
    model = args.model
    max_tokens = DEFAULT_MAX_TOKENS
    max_model_len = DEFAULT_MAX_MODEL_LEN
    extra_flags = ""
    repetition_penalty = None
    temperature = 0
    top_p = None
    min_p = None

    if model in MODEL_OVERRIDES:
        overrides = MODEL_OVERRIDES[model]
        max_tokens = overrides.get("max_tokens", max_tokens)
        max_model_len = overrides.get("max_model_len", max_model_len)
        extra_flags = overrides.get("extra_flags", "")
        repetition_penalty = overrides.get("repetition_penalty")
        temperature = overrides.get("temperature", temperature)   # default 0 preserved
        top_p = overrides.get("top_p")
        min_p = overrides.get("min_p")

    print(f"\n  Model:       {model}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Temperature: {temperature}")
    if repetition_penalty is not None:
        print(f"  Repetition penalty: {repetition_penalty}")
    if top_p is not None:
        print(f"  top_p:       {top_p}")
    if min_p is not None:
        print(f"  min_p:       {min_p}")
    print(f"  Config:      {args.config}")
    print(f"  Container:   {args.container}")

    # --- Checkpoint / resume setup ---
    # The shard namespace AND the consolidated output are both derived from
    # chunk_stem, so each chunk's shards stay isolated (distinct output names
    # per chunk => distinct stems => distinct part_* globs). A --test run gets
    # its own suffixed stem+output so it never resumes from, consolidates, or
    # overwrites a production chunk's shards/output.
    output_path = args.output
    if args.test:
        _root, _ext = os.path.splitext(args.output)
        output_path = f"{_root}.test{args.test}{_ext}"
    results_dir = os.path.dirname(output_path) or "."
    chunk_stem = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(results_dir, exist_ok=True)
    if output_path != args.output:
        print(f"\n  TEST MODE: shards + output isolated -> {output_path}")
    try:
        ckpt_every = max(1, int(os.environ.get("CAMEL_CKPT_EVERY", "25")))
    except ValueError:
        ckpt_every = 25
    completed, start_index, existing_shards = scan_completed_shards(
        results_dir, chunk_stem)
    total_rows_planned = len(chunk_df) * len(
        [p for p in config["prompts"] if p.get("enabled", True)])
    print(f"\n  Checkpoint dir:   {os.path.abspath(results_dir)}")
    print(f"  Checkpoint every: {ckpt_every} texts (CAMEL_CKPT_EVERY)")
    if existing_shards:
        print(f"  resumed: skipped {len(completed)} of {total_rows_planned} "
              f"rows from {len(existing_shards)} shards")
    else:
        print(f"  no existing shards — fresh run")

    # --- Start vLLM (unless --no-server) ---
    vllm_proc = None
    try:
        if args.no_server:
            print(f"\n  --no-server: Using existing vLLM at port {args.port}")
            # Verify it's running
            if not wait_for_vllm(args.port, timeout=10):
                print("ERROR: No vLLM server found. Start one first or remove --no-server.")
                sys.exit(1)
        else:
            print(f"\n  Starting vLLM server...")
            vllm_proc = start_vllm(
                model=model,
                container_path=args.container,
                port=args.port,
                quantization=args.quantization,
                tp=args.tp,
                gpu_util=DEFAULT_GPU_UTIL,
                max_model_len=max_model_len,
                extra_flags=extra_flags,
                log_path=os.path.join(results_dir, f"{chunk_stem}.vllm.log"),
            )

            # 600s was tuned for a single job with the model already warm in the
            # NFS page cache. In a 100+ job production wave, dozens of jobs pull
            # 15-140GB of weights off NetApp at once and load time balloons --
            # the falcon-h1-34b gate (job 676999, 2026-07-30) hit exactly this
            # and was held. Override with CAMEL_VLLM_TIMEOUT if needed.
            _vllm_timeout = int(os.environ.get("CAMEL_VLLM_TIMEOUT", "2400"))
            if not wait_for_vllm(args.port, timeout=_vllm_timeout):
                print("ERROR: vLLM failed to start")
                sys.exit(1)

        # --- Run annotation (flushes durable shards as it goes) ---
        annotate_chunk(
            chunk_df=chunk_df,
            config=config,
            model=model,
            port=args.port,
            max_tokens=max_tokens,
            num_threads=args.threads,
            text_id_col=text_id_col,
            text_col=text_col,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            results_dir=results_dir,
            chunk_stem=chunk_stem,
            ckpt_every=ckpt_every,
            completed=completed,
            start_index=start_index,
        )

        # --- Consolidate shards into the canonical output ---
        shard_paths = shard_glob(results_dir, chunk_stem)
        if not shard_paths:
            print("\n  WARNING: no shards found to consolidate (0 rows?)")
        else:
            shards = [pd.read_feather(p) for p in shard_paths]
            result_df = pd.concat(shards, ignore_index=True)
            # Dedupe on the resumable unit; later shard wins (safety only —
            # completed rows are skipped, so dupes should not occur).
            result_df = result_df.drop_duplicates(
                subset=["text_id", "prompt_id"], keep="last")
            # Deterministic canonical row order for downstream eval.
            if "original_index" in result_df.columns:
                result_df = result_df.sort_values(
                    ["original_index", "prompt_id"], kind="stable")
            else:
                result_df = result_df.sort_values(
                    ["text_id", "prompt_id"], kind="stable")
            # Preserve today's column order exactly: pin to the first shard's
            # column list (concat already inherits it; this makes it explicit
            # and immune to per-shard label-completion-order drift on resume).
            result_df = result_df[list(shards[0].columns)]
            _atomic_write_feather(result_df, output_path)
            print(f"\n  Results consolidated: {output_path}")
            print(f"  Rows: {len(result_df)} (from {len(shard_paths)} shards)")
            print(f"  Columns: {len(result_df.columns)}")

    except KeyboardInterrupt:
        print("\n  INTERRUPTED. Cleaning up...")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if vllm_proc:
            kill_vllm(vllm_proc)

    print("\n" + "=" * 70)
    print("JOB COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
