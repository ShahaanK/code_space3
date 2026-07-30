#!/usr/bin/env python3
"""
select_and_merge.py — consolidate run-versioned CAMEL result files for evaluation.

Apophis-side, read-mostly. This script is the successor to the buggy inline
`python -c "... glob('hpc/results/*.feather') ..."` snippet in
HPC_CAMEL_README.md Step 7, and to the stale `merge_results.py` invocation
that generate_batch_jobs.py still prints (see hpc/generate_batch_jobs.py
~line 592 — NOT updated by this change, flagged as follow-up work).

BACKGROUND — run-versioned naming schema
-----------------------------------------
Production result files land at:

    results_<tag>/results_<tag>_chunk<NNN>_<YYYYMMDD>_run<NNNN>_cl<Cluster>.feather

e.g. results_qwen2.5-72b/results_qwen2.5-72b_chunk000_20260721_run0001_cl667152.feather

Multiple run-versioned files can exist per chunk (re-runs, different
clusters/dates, the OG "subset protocol" of releasing chunk_000 first then
the rest). NOTHING is ever overwritten by the HPC runner — every completed
attempt is preserved on disk as data, so re-runs accumulate rather than
replace.

The runner (camel_annotate_hpc.py) also leaves crash-resume SHARD files in
the same directory, named:

    <basename>.part_NNN.feather

These are intermediate per-N-row checkpoints within a single job attempt and
are NEVER deleted after the job consolidates them into the final
run-versioned file. A naive `*.feather` glob over a results_<tag>/ directory
will therefore double-count rows by including these shards alongside the
consolidated file they were rolled into. This script EXCLUDES them (hard
requirement, see discover_result_files()).

WHAT THIS SCRIPT DOES
----------------------
For a given model's results directory, and for each chunk_id that appears in
that directory:

  1. Find every run-versioned candidate file for that chunk (real result
     files only, shards excluded).
  2. Look up the expected row count for that chunk from the manifest
     (chunk's `rows` field * --expected-prompts prompt arms).
  3. Read each candidate's actual row count and apply a COMPLETENESS GATE:
     only candidates with rows >= expected are eligible for selection. If a
     chunk has zero complete candidates, the chunk is EXCLUDED from the
     merge and reported as missing — an incomplete file is never silently
     selected.
  4. Among complete candidates, select the LATEST by (date, run number)
     descending — i.e. the most recent successful attempt for that chunk.
  5. Log the decision for every chunk: which run was selected, and which
     other (archived, non-selected) run-versioned files were present for
     that chunk, with their row counts and completeness.
  6. Concatenate the selected file per chunk into the merged --output
     Feather file. Print run totals and a final summary of any chunks with
     no complete run.

This script is READ-ONLY with respect to the results_<tag>/ directory: it
never deletes or moves any file (selected or archived). It only writes the
single merged --output file.

DOWNSTREAM
----------
evaluate.py (repo root) consumes results via `--results <path>` (nargs="+"),
so the merged output feeds evaluation directly, e.g.:

    python evaluate.py --results outputs/merged_<tag>.feather \\
        --gold samples/50_random_samples_ans.csv

or, from the hpc/ directory, pointing at whatever --output path you chose.

USAGE
-----
    python select_and_merge.py --results-dir results_qwen2.5-72b
    python select_and_merge.py --results-dir results_llama-3.3-70b \\
        --manifest chunks/chunk_manifest.json --expected-prompts 5 \\
        --output outputs/merged_llama-3.3-70b.feather
"""

import argparse
import json
import os
import re
import sys
from glob import glob

import pandas as pd

# Matches crash-resume shard files left behind by camel_annotate_hpc.py, e.g.
# results_qwen2.5-72b_chunk000_20260721_run0001_cl667152.part_003.feather
PART_SHARD_RE = re.compile(r"\.part_\d+\.feather$")


def build_result_filename_re(tag):
    """Anchored regex for a run-versioned result filename for a given tag.

    Captures (chunk_id, date, run_no, cluster_id) as strings.
    Schema: results_<tag>_chunk<NNN>_<YYYYMMDD>_run<NNNN>_cl<Cluster>.feather
    """
    escaped_tag = re.escape(tag)
    pattern = (
        rf"^results_{escaped_tag}_chunk(\d+)_(\d{{8}})_run(\d+)_cl(\d+)\.feather$"
    )
    return re.compile(pattern)


def derive_tag(results_dir, explicit_tag):
    """Derive the model tag from --tag, else from the results-dir basename
    by stripping a leading 'results_'."""
    if explicit_tag:
        return explicit_tag
    base = os.path.basename(os.path.normpath(results_dir))
    if base.startswith("results_"):
        return base[len("results_"):]
    return base


def discover_result_files(results_dir):
    """Glob all .feather files in results_dir, excluding crash-resume shards
    (basename matching .part_NNN.feather). This exclusion is a HARD
    requirement -- shards double-count rows if merged alongside the
    consolidated file they were rolled into."""
    all_feathers = sorted(glob(os.path.join(results_dir, "*.feather")))
    kept, shards = [], []
    for path in all_feathers:
        if PART_SHARD_RE.search(os.path.basename(path)):
            shards.append(path)
        else:
            kept.append(path)
    return kept, shards


def parse_candidates(files, name_re):
    """Parse each filename against the anchored run-versioned schema regex.
    Returns (candidates, unmatched) where candidates is a list of dicts:
    {path, chunk_id (int), date (str), run_no (int), cluster_id (str)}.
    Files that don't match are returned in `unmatched` for warning/logging.
    """
    candidates = []
    unmatched = []
    for path in files:
        base = os.path.basename(path)
        m = name_re.match(base)
        if not m:
            unmatched.append(path)
            continue
        chunk_id_str, date_str, run_no_str, cluster_id = m.groups()
        candidates.append({
            "path": path,
            "chunk_id": int(chunk_id_str),
            "date": date_str,
            "run_no": int(run_no_str),
            "cluster_id": cluster_id,
        })
    return candidates, unmatched


def load_manifest_rows(manifest_path):
    """Load chunks/chunk_manifest.json and return {chunk_id (int): rows (int)}."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    rows_by_chunk = {}
    for entry in manifest.get("chunks", []):
        rows_by_chunk[int(entry["chunk_id"])] = int(entry["rows"])
    return rows_by_chunk


def label_candidate(cand):
    """Short label for logging: run<NNNN>_cl<Cluster>."""
    return f"run{cand['run_no']:04d}_cl{cand['cluster_id']}"


def select_per_chunk(candidates_by_chunk, rows_by_chunk, expected_prompts):
    """For each chunk_id, apply the completeness gate and pick the latest
    complete candidate by (date, run_no) descending.

    Returns (selections, excluded_chunks, log_lines) where:
      - selections: {chunk_id: {selected_cand, actual_rows}}
      - excluded_chunks: list of chunk_ids with zero complete candidates
      - log_lines: list of str, one (or more) per chunk, decision log
    """
    selections = {}
    excluded_chunks = []
    log_lines = []

    for chunk_id in sorted(candidates_by_chunk.keys()):
        cands = candidates_by_chunk[chunk_id]

        if chunk_id not in rows_by_chunk:
            log_lines.append(
                f"chunk{chunk_id:03d}: WARNING - chunk_id not found in manifest, "
                f"skipping ({len(cands)} candidate file(s) present)"
            )
            excluded_chunks.append(chunk_id)
            continue

        expected_rows = rows_by_chunk[chunk_id] * expected_prompts

        # Read actual row counts for every candidate.
        enriched = []
        for c in cands:
            try:
                actual_rows = len(pd.read_feather(c["path"]))
            except Exception as exc:
                print(f"  WARNING: failed to read {c['path']}: {exc}", file=sys.stderr)
                continue
            enriched.append({**c, "actual_rows": actual_rows,
                              "complete": actual_rows >= expected_rows})

        if not enriched:
            log_lines.append(
                f"chunk{chunk_id:03d}: WARNING - no readable candidate files "
                f"(expected {expected_rows} rows) -- EXCLUDED from merge"
            )
            excluded_chunks.append(chunk_id)
            continue

        complete = [c for c in enriched if c["complete"]]

        if not complete:
            archived_desc = ", ".join(
                f"{label_candidate(c)} ({c['actual_rows']} rows, incomplete)"
                for c in sorted(enriched, key=lambda c: (c["date"], c["run_no"]), reverse=True)
            )
            log_lines.append(
                f"chunk{chunk_id:03d}: WARNING - NO complete run "
                f"(expected {expected_rows} rows); candidates present: "
                f"{archived_desc} -- EXCLUDED from merge"
            )
            excluded_chunks.append(chunk_id)
            continue

        # Latest by (date, run_no) descending among complete candidates.
        complete_sorted = sorted(
            complete, key=lambda c: (c["date"], c["run_no"]), reverse=True
        )
        selected = complete_sorted[0]

        # Archived = every other candidate (complete or not), most-recent first.
        archived = [c for c in enriched if c["path"] != selected["path"]]
        archived_sorted = sorted(
            archived, key=lambda c: (c["date"], c["run_no"]), reverse=True
        )

        if archived_sorted:
            archived_desc = ", ".join(
                f"{label_candidate(c)} ({c['actual_rows']} rows"
                f"{'' if c['complete'] else ', incomplete'})"
                for c in archived_sorted
            )
            log_lines.append(
                f"chunk{chunk_id:03d}: selected {label_candidate(selected)} "
                f"({selected['actual_rows']} rows); archived also-present: {archived_desc}"
            )
        else:
            log_lines.append(
                f"chunk{chunk_id:03d}: selected {label_candidate(selected)} "
                f"({selected['actual_rows']} rows); no other runs present"
            )

        selections[chunk_id] = selected

    return selections, excluded_chunks, log_lines


def main():
    parser = argparse.ArgumentParser(
        description="Select the latest complete run-versioned CAMEL result "
                     "file per chunk and merge into a single Feather file "
                     "for evaluate.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory of run-versioned result files, e.g. results_qwen2.5-72b",
    )
    parser.add_argument(
        "--manifest", default="chunks/chunk_manifest.json",
        help="Path to chunk_manifest.json (default: chunks/chunk_manifest.json)",
    )
    parser.add_argument(
        "--expected-prompts", type=int, default=5,
        help="Number of prompt arms enabled in the config used for this run "
             "(default: 5, the OG production config3.yaml count). Do NOT "
             "read Apophis's config3.yaml for this -- it enables only 3.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Merged output Feather path (default: merged_<tag>.feather)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Model tag (default: derived from results-dir basename by "
             "stripping a leading 'results_')",
    )
    args = parser.parse_args()

    tag = derive_tag(args.results_dir, args.tag)
    output_path = args.output or f"merged_{tag}.feather"

    print(f"Results dir:      {args.results_dir}")
    print(f"Tag:              {tag}")
    print(f"Manifest:         {args.manifest}")
    print(f"Expected prompts: {args.expected_prompts}")
    print(f"Output:           {output_path}")
    print()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results-dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    kept_files, shard_files = discover_result_files(args.results_dir)
    if shard_files:
        print(f"Excluded {len(shard_files)} crash-resume shard file(s) (.part_NNN.feather):")
        for s in shard_files:
            print(f"  - {s}")
        print()

    name_re = build_result_filename_re(tag)
    candidates, unmatched = parse_candidates(kept_files, name_re)

    if unmatched:
        print(f"WARNING: {len(unmatched)} file(s) did not match the "
              f"run-versioned schema for tag '{tag}' and were skipped:")
        for u in unmatched:
            print(f"  - {u}")
        print()

    if not candidates:
        print("No run-versioned candidate files found. Nothing to merge.")
        sys.exit(0)

    if not os.path.isfile(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)
    rows_by_chunk = load_manifest_rows(args.manifest)

    candidates_by_chunk = {}
    for c in candidates:
        candidates_by_chunk.setdefault(c["chunk_id"], []).append(c)

    selections, excluded_chunks, log_lines = select_per_chunk(
        candidates_by_chunk, rows_by_chunk, args.expected_prompts
    )

    print("Per-chunk selection log:")
    for line in log_lines:
        print(f"  {line}")
    print()

    if not selections:
        print("No chunks had a complete run. Nothing to merge.")
        if excluded_chunks:
            print(f"Chunks with NO complete run: "
                  f"{', '.join(f'chunk{c:03d}' for c in sorted(excluded_chunks))}")
        sys.exit(0)

    frames = []
    for chunk_id in sorted(selections.keys()):
        path = selections[chunk_id]["path"]
        frames.append(pd.read_feather(path))

    merged = pd.concat(frames, ignore_index=True)
    merged.to_feather(output_path)

    print("=" * 70)
    print(f"Chunks selected:  {len(selections)}")
    print(f"Chunks excluded:  {len(excluded_chunks)}")
    print(f"Total rows:       {len(merged)}")
    print(f"Merged output:    {output_path}")
    print("=" * 70)

    if excluded_chunks:
        print()
        print("Chunks with NO complete run (excluded from merge -- MISSING data):")
        for c in sorted(excluded_chunks):
            print(f"  - chunk{c:03d}")


if __name__ == "__main__":
    main()
