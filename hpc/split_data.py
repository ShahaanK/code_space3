#!/usr/bin/env python3
"""
Data Splitting Utility for HPC Belief Extraction

This script splits the remaining unprocessed data into chunks for parallel
processing on an HPC cluster.

It reads the current extraction checkpoint, identifies unprocessed rows,
and splits them into chunks of specified size.

Usage:
    # Create test chunk (1000 rows from unprocessed data)
    python split_data.py --create-test --test-size 1000

    # Split remaining data into chunks
    python split_data.py --chunk-size 250000

    # Custom paths
    python split_data.py --input /path/to/input.feather \\
                         --checkpoint /path/to/checkpoint.feather \\
                         --output-dir chunks/
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys


def load_checkpoint_progress(checkpoint_path: Path) -> set:
    """
    Load checkpoint and return set of indices that have been processed.

    A row is considered processed if it has a non-empty beliefs JSON array.
    """
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return set()

    print(f"Loading checkpoint: {checkpoint_path}")
    df = pd.read_feather(checkpoint_path)

    processed_indices = set()
    for idx, beliefs_json in enumerate(df['beliefs']):
        try:
            if isinstance(beliefs_json, str) and beliefs_json != '[]':
                beliefs = json.loads(beliefs_json)
                if beliefs:  # Non-empty list
                    processed_indices.add(idx)
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"  Found {len(processed_indices):,} processed rows")
    return processed_indices


def split_data(
    input_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    chunk_size: int = 250000,
    create_test: bool = False,
    test_size: int = 1000,
):
    """
    Split unprocessed data into chunks.

    Args:
        input_path: Path to original input feather file
        checkpoint_path: Path to extraction checkpoint
        output_dir: Directory for output chunks
        chunk_size: Number of rows per chunk
        create_test: If True, only create a test chunk
        test_size: Size of test chunk
    """
    print("=" * 60)
    print("Data Splitting Utility")
    print("=" * 60)

    # Load original input data
    print(f"\nLoading input data: {input_path}")
    df = pd.read_feather(input_path)
    print(f"  Total rows: {len(df):,}")

    # Load checkpoint and find processed indices
    if checkpoint_path is not None:
        processed_indices = load_checkpoint_progress(checkpoint_path)
    else:
        print("No checkpoint provided — treating all rows as unprocessed")
        processed_indices = set()

    # Find unprocessed rows
    all_indices = set(range(len(df)))
    unprocessed_indices = sorted(all_indices - processed_indices)
    print(f"\nUnprocessed rows: {len(unprocessed_indices):,}")
    print(f"  ({len(unprocessed_indices)/len(df)*100:.1f}% remaining)")

    if len(unprocessed_indices) == 0:
        print("All rows have been processed!")
        return

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if create_test:
        # Create single test chunk
        test_indices = unprocessed_indices[:test_size]
        test_df = df.iloc[test_indices].copy()
        test_df['original_index'] = test_indices  # Preserve original index for merging

        test_path = output_dir / 'test_chunk.feather'
        test_df.to_feather(test_path)
        print(f"\nCreated test chunk: {test_path}")
        print(f"  Rows: {len(test_df):,}")
        print(f"  Original indices: {test_indices[0]} - {test_indices[-1]}")
    else:
        # Split into production chunks
        num_chunks = (len(unprocessed_indices) + chunk_size - 1) // chunk_size
        print(f"\nSplitting into {num_chunks} chunks of ~{chunk_size:,} rows each")

        chunk_manifest = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(unprocessed_indices))
            chunk_indices = unprocessed_indices[start_idx:end_idx]

            chunk_df = df.iloc[chunk_indices].copy()
            chunk_df['original_index'] = chunk_indices  # Preserve for merging

            chunk_name = f"chunk_{i:03d}.feather"
            chunk_path = output_dir / chunk_name
            chunk_df.to_feather(chunk_path)

            chunk_info = {
                'chunk_id': i,
                'filename': chunk_name,
                'rows': len(chunk_df),
                'original_start_index': chunk_indices[0],
                'original_end_index': chunk_indices[-1],
            }
            chunk_manifest.append(chunk_info)

            print(f"  Created {chunk_name}: {len(chunk_df):,} rows "
                  f"(indices {chunk_indices[0]:,} - {chunk_indices[-1]:,})")

        # Save manifest
        manifest_path = output_dir / 'chunk_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump({
                'total_chunks': num_chunks,
                'chunk_size': chunk_size,
                'total_unprocessed': len(unprocessed_indices),
                'chunks': chunk_manifest,
            }, f, indent=2)
        print(f"\nSaved manifest: {manifest_path}")



def main():
    parser = argparse.ArgumentParser(
        description='Split data for HPC belief extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Path to input feather file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to extraction checkpoint (optional — if omitted, all rows are treated as unprocessed)')
    parser.add_argument('--output-dir', type=str,
                       default='./chunks',
                       help='Output directory for chunks (default: ./chunks)')

    parser.add_argument('--chunk-size', type=int, default=250000,
                       help='Number of rows per chunk (default: 250000)')

    parser.add_argument('--create-test', action='store_true',
                       help='Create only a test chunk')
    parser.add_argument('--test-size', type=int, default=1000,
                       help='Size of test chunk (default: 1000)')

    args = parser.parse_args()

    # Resolve all paths relative to CWD (standard CLI behavior)
    input_path = Path(args.input).resolve()

    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).resolve()

    output_dir = Path(args.output_dir).resolve()

    # Validate inputs
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    split_data(
        input_path=input_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        create_test=args.create_test,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
