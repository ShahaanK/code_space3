#!/usr/bin/env python3
"""
CAMEL Annotation — Merge HPC Results
======================================
Combines individual chunk results from HPC jobs into a single Feather file.

After all HTCondor jobs complete, run this to merge:
    python merge_results.py

Options:
    python merge_results.py --results-dir results/      # custom results dir
    python merge_results.py --output merged_results.feather
    python merge_results.py --check                      # check completeness only

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def find_result_files(results_dir):
    """Find all result Feather files in the results directory."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("results_*.feather"))
    return files


def load_manifest(chunks_dir="chunks"):
    """Load the chunk manifest to check completeness."""
    manifest_path = Path(chunks_dir) / "chunk_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge HPC chunk results into a single file")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result Feather files")
    parser.add_argument("--chunks-dir", default="chunks",
                        help="Directory containing chunk manifest")
    parser.add_argument("--output", default="merged_annotation_results.feather",
                        help="Output merged file path")
    parser.add_argument("--check", action="store_true",
                        help="Only check completeness, don't merge")
    args = parser.parse_args()

    print("=" * 60)
    print("CAMEL Annotation — Merge HPC Results")
    print("=" * 60)

    # --- Find result files ---
    result_files = find_result_files(args.results_dir)
    print(f"\n  Results dir: {args.results_dir}")
    print(f"  Result files found: {len(result_files)}")

    if not result_files:
        print(f"\n  ERROR: No result files found in {args.results_dir}/")
        print(f"  Expected files matching: results_*.feather")
        sys.exit(1)

    # --- Check against manifest ---
    manifest = load_manifest(args.chunks_dir)
    if manifest:
        expected = manifest["total_chunks"]
        found = len(result_files)
        missing = expected - found

        print(f"\n  Manifest: {expected} chunks expected")
        print(f"  Found:    {found} result files")

        if missing > 0:
            # Identify which chunks are missing
            expected_ids = set(range(expected))
            found_ids = set()
            for f in result_files:
                try:
                    chunk_id = int(f.stem.split("_")[1])
                    found_ids.add(chunk_id)
                except (IndexError, ValueError):
                    pass

            missing_ids = sorted(expected_ids - found_ids)
            print(f"  MISSING:  {missing} chunks: {missing_ids}")

            if args.check:
                sys.exit(1)

            print(f"\n  WARNING: Merging incomplete results ({found}/{expected})")
        else:
            print(f"  Status:   ALL COMPLETE")

            if args.check:
                print("\n  All chunks processed successfully.")
                sys.exit(0)
    else:
        print(f"\n  No manifest found — merging all available results")

    if args.check:
        sys.exit(0)

    # --- Load and merge ---
    print(f"\n  Loading result files...")
    dfs = []
    total_rows = 0

    for f in result_files:
        try:
            df = pd.read_feather(f)
            dfs.append(df)
            total_rows += len(df)
            print(f"    {f.name}: {len(df)} rows")
        except Exception as e:
            print(f"    {f.name}: ERROR — {e}")

    if not dfs:
        print("\n  ERROR: No valid result files could be loaded")
        sys.exit(1)

    # Concatenate
    merged = pd.concat(dfs, ignore_index=True)

    # Sort by text_id and prompt_id for cleanliness
    sort_cols = []
    if "text_id" in merged.columns:
        sort_cols.append("text_id")
    if "prompt_id" in merged.columns:
        sort_cols.append("prompt_id")
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # --- Check for duplicates ---
    if "text_id" in merged.columns and "prompt_id" in merged.columns:
        dup_cols = ["text_id", "prompt_id", "model"]
        dup_cols = [c for c in dup_cols if c in merged.columns]
        duplicates = merged.duplicated(subset=dup_cols, keep=False)
        if duplicates.any():
            n_dups = duplicates.sum()
            print(f"\n  WARNING: {n_dups} duplicate rows detected")
            print(f"  Keeping first occurrence, dropping {n_dups // 2} duplicates")
            merged = merged.drop_duplicates(subset=dup_cols, keep="first")

    # --- Save ---
    merged.reset_index(drop=True).to_feather(args.output)

    print(f"\n  Merged results saved: {args.output}")
    print(f"  Total rows:  {len(merged)}")
    print(f"  Columns:     {len(merged.columns)}")
    print(f"  File size:   {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

    # --- Summary stats ---
    if "model" in merged.columns:
        print(f"\n  Models:")
        for model, count in merged["model"].value_counts().items():
            print(f"    {model}: {count} rows")

    if "prompt_name" in merged.columns:
        print(f"\n  Prompts:")
        for prompt, count in merged["prompt_name"].value_counts().items():
            print(f"    {prompt}: {count} rows")

    unique_texts = merged["text_id"].nunique() if "text_id" in merged.columns else "?"
    print(f"\n  Unique texts: {unique_texts}")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
