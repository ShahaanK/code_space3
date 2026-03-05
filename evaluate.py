#!/usr/bin/env python3
"""
CAMEL Annotation Pipeline — Evaluation Module
================================================
Standalone script that compares pipeline predictions against the human
gold standard and computes per-label, per-model, and per-prompt metrics.

Usage:
    python evaluate.py                                    # auto-find latest results
    python evaluate.py --results outputs/camel_*.csv      # specific result files
    python evaluate.py --gold sample_prompts_human_key.xlsx
    python evaluate.py --approach strict                  # strict majority (default)
    python evaluate.py --approach lenient                 # any annotator = positive
    python evaluate.py --output eval_report.csv           # save detailed report

Gold Standard Format (sample_prompts_human_key.xlsx):
    Columns: Number, [25 label columns with 0/1 values]
    Where 1 = label present (majority agreement), 0 = absent

Results Format (pipeline CSV output):
    Columns: text_id, prompt_id, prompt_name, model, provider, ...,
             [25 label columns with 1/0/-1 values], [25 response columns]
    Where 1 = YES, 0 = NO, -1 = UNCLEAR/ERROR

Author: Shahaan Khan
Research: Prof. Joshua Introne, Syracuse University iSchool
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np


# All 25 CAMEL labels in standard order
ALL_LABELS = [
    "Individualism", "Collectivism", "Honor", "Tightness", "Looseness",
    "Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity",
    "Ownership", "Liberty", "Honesty", "ThinMorality",
    "OffensiveUncivilLanguage", "Hate", "Fear", "Threat",
    "CreativityInnovation", "Kinship", "Religion", "IntellectualHumility",
    "AnalyticalThinking", "GeneralizedTrust"
]


# =============================================================================
# GOLD STANDARD LOADING
# =============================================================================

def load_gold_standard(gold_path, approach="strict"):
    """
    Load and binarize the human gold standard.

    Args:
        gold_path: Path to the gold standard Excel file.
        approach: "strict" (only majority=1 counts) or "lenient" (1 and -1 count).

    Returns:
        DataFrame with columns: text_id + 25 binary label columns (0/1).
    """
    df = pd.read_excel(gold_path) if str(gold_path).endswith((".xlsx", ".xls")) else pd.read_csv(gold_path)

    # Detect format: could be "Number" or "Text Number" as ID column
    if "Number" in df.columns:
        id_col = "Number"
    elif "Text Number" in df.columns:
        id_col = "Text Number"
    else:
        raise ValueError(f"Gold standard must have 'Number' or 'Text Number' column. "
                         f"Found: {df.columns.tolist()}")

    # Find which label columns exist
    label_cols = [c for c in ALL_LABELS if c in df.columns]
    if not label_cols:
        raise ValueError(f"No label columns found in gold standard. "
                         f"Expected columns like: {ALL_LABELS[:5]}")

    gold = df[[id_col]].copy()
    gold = gold.rename(columns={id_col: "text_id"})

    for col in label_cols:
        if approach == "strict":
            # Only majority agreement (value == 1) counts as positive
            gold[col] = (df[col] == 1).astype(int)
        elif approach == "lenient":
            # Any annotator mention (value == 1 or -1) counts as positive
            gold[col] = (df[col].isin([1, -1])).astype(int)
        else:
            raise ValueError(f"Unknown approach: {approach}. Use 'strict' or 'lenient'.")

    # Add any missing labels as all-zero columns
    for col in ALL_LABELS:
        if col not in gold.columns:
            gold[col] = 0

    return gold, label_cols


# =============================================================================
# RESULTS LOADING
# =============================================================================

def load_results(results_paths):
    """
    Load and concatenate one or more pipeline result CSVs.

    Returns:
        DataFrame with all results, filtered to valid predictions only.
    """
    dfs = []
    for path in results_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(path)}")
        except Exception as e:
            print(f"  WARNING: Could not load {path}: {e}")

    if not dfs:
        raise ValueError("No result files could be loaded.")

    results = pd.concat(dfs, ignore_index=True)

    # Filter out rows where model is Ollama (connection errors)
    ollama_mask = results["provider"] == "local_ollama"
    if ollama_mask.any():
        print(f"  Filtered out {ollama_mask.sum()} Ollama rows (connection errors)")
        results = results[~ollama_mask]

    return results


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_binary_metrics(y_true, y_pred):
    """
    Compute precision, recall, F1 for a single label.

    Args:
        y_true: array of gold labels (0/1)
        y_pred: array of predictions (0/1, -1 treated as 0)

    Returns:
        dict with tp, fp, fn, tn, precision, recall, f1, support
    """
    # Treat -1 (unclear/error) as 0 (negative prediction)
    y_pred = np.where(y_pred == -1, 0, y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,  # number of actual positives
        "error_count": int((y_pred == -1).sum()) if hasattr(y_pred, '__len__') else 0,
    }


def evaluate_group(results_group, gold, label_cols):
    """
    Evaluate a group of predictions (e.g., one model+prompt combo) against gold.

    Args:
        results_group: DataFrame with predictions for a specific model/prompt.
        gold: DataFrame with gold standard labels.
        label_cols: List of label column names to evaluate.

    Returns:
        dict with per-label metrics and macro averages.
    """
    # Merge predictions with gold on text_id
    merged = results_group.merge(
        gold, on="text_id", how="inner", suffixes=("_pred", "_gold")
    )

    if len(merged) == 0:
        return None

    label_metrics = {}
    f1_scores = []
    labels_with_support = 0

    for label in label_cols:
        pred_col = f"{label}_pred" if f"{label}_pred" in merged.columns else label
        gold_col = f"{label}_gold" if f"{label}_gold" in merged.columns else label

        if pred_col not in merged.columns or gold_col not in merged.columns:
            continue

        y_true = merged[gold_col].values
        y_pred = merged[pred_col].values

        metrics = compute_binary_metrics(y_true, y_pred)
        label_metrics[label] = metrics

        # Only include labels with at least one positive in gold for macro-F1
        if metrics["support"] > 0:
            f1_scores.append(metrics["f1"])
            labels_with_support += 1

    # Macro averages (only over labels that have positive examples in gold)
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    macro_precision = np.mean([m["precision"] for m in label_metrics.values()
                               if m["support"] > 0]) if labels_with_support > 0 else 0.0
    macro_recall = np.mean([m["recall"] for m in label_metrics.values()
                            if m["support"] > 0]) if labels_with_support > 0 else 0.0

    # Micro averages (over all labels)
    total_tp = sum(m["tp"] for m in label_metrics.values())
    total_fp = sum(m["fp"] for m in label_metrics.values())
    total_fn = sum(m["fn"] for m in label_metrics.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    return {
        "per_label": label_metrics,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "texts_evaluated": len(merged),
        "labels_with_support": labels_with_support,
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(model, prompt_name, eval_result):
    """Print a formatted summary for one model+prompt combination."""
    print(f"\n  Model: {model}")
    print(f"  Prompt: {prompt_name}")
    print(f"  Texts evaluated: {eval_result['texts_evaluated']}")
    print(f"  Labels with gold positives: {eval_result['labels_with_support']}/25")
    print(f"  Macro F1:  {eval_result['macro_f1']:.3f}  "
          f"(P={eval_result['macro_precision']:.3f}, R={eval_result['macro_recall']:.3f})")
    print(f"  Micro F1:  {eval_result['micro_f1']:.3f}  "
          f"(P={eval_result['micro_precision']:.3f}, R={eval_result['micro_recall']:.3f})")


def print_label_table(eval_result, label_cols):
    """Print per-label metrics as a table."""
    print(f"\n  {'Label':<28s} {'P':>6s} {'R':>6s} {'F1':>6s} "
          f"{'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} {'Sup':>4s}")
    print(f"  {'-'*76}")

    for label in label_cols:
        if label in eval_result["per_label"]:
            m = eval_result["per_label"][label]
            marker = " *" if m["support"] == 0 else ""
            print(f"  {label:<28s} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d} "
                  f"{m['tn']:>4d} {m['support']:>4d}{marker}")

    print(f"\n  * = no positive examples in gold (F1 undefined, excluded from macro)")


def build_detail_rows(results, gold, label_cols):
    """
    Build a list of dicts for the detailed CSV report.
    One row per model × prompt × label.
    """
    rows = []
    groups = results.groupby(["model", "prompt_id", "prompt_name"])

    for (model, prompt_id, prompt_name), group in groups:
        eval_result = evaluate_group(group, gold, label_cols)
        if eval_result is None:
            continue

        for label in label_cols:
            if label not in eval_result["per_label"]:
                continue
            m = eval_result["per_label"][label]
            rows.append({
                "model": model,
                "prompt_id": prompt_id,
                "prompt_name": prompt_name,
                "label": label,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
                "support": m["support"],
                "macro_f1": eval_result["macro_f1"],
                "micro_f1": eval_result["micro_f1"],
                "texts_evaluated": eval_result["texts_evaluated"],
            })

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CAMEL Evaluation Module")
    parser.add_argument("--results", nargs="+", default=None,
                        help="Path(s) to result CSV files. Default: auto-find in outputs/")
    parser.add_argument("--gold", default="sample_prompts_human_key.xlsx",
                        help="Path to gold standard Excel file")
    parser.add_argument("--raw-gold", default=None,
                        help="Path to raw annotator file (sample_human_annotators_ans.xlsx). "
                             "If provided, gold standard is computed from this file instead.")
    parser.add_argument("--approach", default="strict",
                        choices=["strict", "lenient"],
                        help="How to binarize gold labels: "
                             "'strict' = majority only (default), "
                             "'lenient' = any annotator mention")
    parser.add_argument("--output", default=None,
                        help="Save detailed per-label report to CSV")
    parser.add_argument("--labels-only", action="store_true",
                        help="Only show labels with positive gold examples")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-label breakdown for each model+prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("CAMEL Evaluation Module")
    print("=" * 70)

    # --- Load gold standard ---
    if args.raw_gold:
        print(f"\nLoading raw annotator data: {args.raw_gold}")
        print(f"  Approach: {args.approach}")
        gold, label_cols = load_gold_standard(args.raw_gold, approach=args.approach)
    else:
        print(f"\nLoading gold standard: {args.gold}")
        print(f"  Approach: {args.approach}")
        gold, label_cols = load_gold_standard(args.gold, approach=args.approach)

    total_positives = gold[label_cols].sum().sum()
    total_cells = len(gold) * len(label_cols)
    print(f"  Texts: {len(gold)}")
    print(f"  Labels: {len(label_cols)}")
    print(f"  Positive labels: {total_positives}/{total_cells} "
          f"({100*total_positives/total_cells:.1f}%)")

    # Show which labels have no positive examples
    zero_support = [c for c in label_cols if gold[c].sum() == 0]
    if zero_support:
        print(f"  Labels with zero positives (unevaluable): {zero_support}")

    # --- Load results ---
    if args.results:
        result_paths = args.results
    else:
        # Auto-find latest results in outputs/
        result_paths = sorted(glob.glob("outputs/camel_results_fullguide_*.csv"))
        if not result_paths:
            result_paths = sorted(glob.glob("outputs/camel_results_*.csv"))
        if not result_paths:
            print("\nERROR: No result files found in outputs/. "
                  "Specify with --results.")
            return

    print(f"\nLoading results:")
    results = load_results(result_paths)
    print(f"  Total rows: {len(results)}")

    # --- Check for text_id overlap ---
    gold_ids = set(gold["text_id"].values)
    result_ids = set(results["text_id"].values)
    overlap = gold_ids & result_ids
    if not overlap:
        print(f"\nERROR: No overlapping text IDs between gold and results!")
        print(f"  Gold IDs: {sorted(gold_ids)}")
        print(f"  Result IDs: {sorted(result_ids)[:20]}...")
        return
    print(f"  Overlapping text IDs: {len(overlap)}/{len(gold_ids)}")

    # --- Evaluate per model × prompt ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    groups = results.groupby(["model", "prompt_id", "prompt_name"])
    all_summaries = []

    for (model, prompt_id, prompt_name), group in groups:
        eval_result = evaluate_group(group, gold, label_cols)
        if eval_result is None:
            print(f"\n  {model} / {prompt_name}: No matching texts")
            continue

        print_summary(model, prompt_name, eval_result)

        if args.verbose:
            eval_labels = label_cols
            if args.labels_only:
                eval_labels = [c for c in label_cols if gold[c].sum() > 0]
            print_label_table(eval_result, eval_labels)

        all_summaries.append({
            "model": model,
            "prompt_id": prompt_id,
            "prompt_name": prompt_name,
            "macro_f1": eval_result["macro_f1"],
            "macro_precision": eval_result["macro_precision"],
            "macro_recall": eval_result["macro_recall"],
            "micro_f1": eval_result["micro_f1"],
            "texts": eval_result["texts_evaluated"],
        })

    # --- Summary table ---
    if all_summaries:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"\n  {'Model':<50s} {'Prompt':<30s} {'F1-M':>6s} {'P-M':>6s} "
              f"{'R-M':>6s} {'F1-m':>6s}")
        print(f"  {'-'*110}")
        for s in all_summaries:
            short_model = s["model"].split("/")[-1][:48]
            print(f"  {short_model:<50s} {s['prompt_name']:<30s} "
                  f"{s['macro_f1']:>6.3f} {s['macro_precision']:>6.3f} "
                  f"{s['macro_recall']:>6.3f} {s['micro_f1']:>6.3f}")

        print(f"\n  F1-M = Macro F1, P-M = Macro Precision, R-M = Macro Recall, "
              f"F1-m = Micro F1")

    # --- Save detailed report ---
    if args.output:
        detail_rows = build_detail_rows(results, gold, label_cols)
        if detail_rows:
            report_df = pd.DataFrame(detail_rows)
            report_df.to_csv(args.output, index=False)
            print(f"\n  Detailed report saved to: {args.output}")
    else:
        # Auto-save to outputs/
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_path = f"outputs/eval_report_{timestamp}.csv"
        detail_rows = build_detail_rows(results, gold, label_cols)
        if detail_rows:
            report_df = pd.DataFrame(detail_rows)
            report_df.to_csv(auto_path, index=False)
            print(f"\n  Detailed report saved to: {auto_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
