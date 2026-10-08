from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGE_READY = ROOT / "data" / "processed" / "active_learning" / "active_learning_merge_ready_160.csv"
DEFAULT_WORKBOOK = ROOT / "data" / "processed" / "active_learning" / "active_learning_merge_holdout_2_adj_corr_470.xlsx"
DEFAULT_OUTPUT_FINAL = ROOT / "data" / "processed" / "active_learning" / "active_learning_merge_final_159.csv"
DEFAULT_OUTPUT_EXCLUDED = ROOT / "data" / "processed" / "active_learning" / "active_learning_merge_excluded_after_resolution.csv"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "active_learning" / "active_learning_holdout_resolution_manifest.json"
PRIMARY_FIELDS = [
    "relevance",
    "diseases_quick",
    "diseases_quick_other_text",
    "has_relevant_5c",
    "disease_1",
    "disease_1_other_text",
    "c5_presence_d1",
    "valence_confidence_d1",
    "valence_complacency_d1",
    "valence_constraints_d1",
    "valence_calculation_d1",
    "valence_collective_d1",
    "annotation_notes",
]
VALENCE_FIELDS = [
    "valence_confidence_d1",
    "valence_complacency_d1",
    "valence_constraints_d1",
    "valence_calculation_d1",
    "valence_collective_d1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a manual resolution for AL duplicate holdout rows.")
    parser.add_argument("--merge-ready", type=Path, default=DEFAULT_MERGE_READY)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet", default="active_learning_merge_holdout_2")
    parser.add_argument("--keep-task-id", default="470")
    parser.add_argument("--output-final", type=Path, default=DEFAULT_OUTPUT_FINAL)
    parser.add_argument("--output-excluded", type=Path, default=DEFAULT_OUTPUT_EXCLUDED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def normalize_valence(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in {"+1", "0", "-1"}:
        return text
    if text == "1":
        return "+1"
    if text == "-1":
        return "-1"
    if text == "0":
        return "0"
    return text


def main() -> None:
    args = parse_args()
    merge_ready = pd.read_csv(args.merge_ready, dtype=str, keep_default_na=False)
    workbook = pd.read_excel(args.workbook, sheet_name=args.sheet, dtype=str).fillna("")

    keep_task_id = str(args.keep_task_id)
    if "labelstudio_task_id" not in workbook.columns:
        raise ValueError("Workbook must contain labelstudio_task_id.")

    workbook["labelstudio_task_id"] = workbook["labelstudio_task_id"].astype(str)
    merge_ready["labelstudio_task_id"] = merge_ready["labelstudio_task_id"].astype(str)

    selected = workbook[workbook["labelstudio_task_id"] == keep_task_id]
    if selected.empty:
        raise ValueError(f"Could not find keep task id {keep_task_id} in workbook.")
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one workbook row for keep task id {keep_task_id}.")
    selected_row = selected.iloc[0]

    cluster_signature = str(
        merge_ready.loc[merge_ready["labelstudio_task_id"] == keep_task_id, "qa_training_dedup_signature"].iloc[0]
    )
    cluster_mask = merge_ready["qa_training_dedup_signature"] == cluster_signature
    cluster_rows = merge_ready.loc[cluster_mask].copy()
    if cluster_rows.empty:
        raise ValueError(f"No duplicate cluster found for keep task id {keep_task_id}.")

    for field in PRIMARY_FIELDS:
        if field in merge_ready.columns and field in workbook.columns:
            value = str(selected_row.get(field, "") or "")
            if field in VALENCE_FIELDS:
                value = normalize_valence(value)
            merge_ready.loc[merge_ready["labelstudio_task_id"] == keep_task_id, field] = value

    keep_mask = merge_ready["labelstudio_task_id"] == keep_task_id
    drop_mask = cluster_mask & ~keep_mask

    merge_ready.loc[keep_mask, "training_dedup_disagreement"] = "resolved_manual_keep"
    merge_ready.loc[keep_mask, "merge_exclusion_reason"] = ""
    merge_ready.loc[keep_mask, "recommended_keep_for_train_dev_merge"] = "yes"
    merge_ready.loc[drop_mask, "training_dedup_disagreement"] = "resolved_manual_drop"
    merge_ready.loc[drop_mask, "merge_exclusion_reason"] = f"manual_duplicate_resolution_drop_in_favor_of_{keep_task_id}"
    merge_ready.loc[drop_mask, "recommended_keep_for_train_dev_merge"] = "no"

    final_df = merge_ready[merge_ready["recommended_keep_for_train_dev_merge"] == "yes"].copy()
    excluded_df = merge_ready[merge_ready["recommended_keep_for_train_dev_merge"] != "yes"].copy()

    args.output_final.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_final, index=False, encoding="utf-8-sig")
    excluded_df.to_csv(args.output_excluded, index=False, encoding="utf-8-sig")

    manifest = {
        "merge_ready_input": str(args.merge_ready),
        "resolution_workbook": str(args.workbook),
        "resolution_sheet": args.sheet,
        "keep_task_id": keep_task_id,
        "resolved_cluster_size": int(cluster_mask.sum()),
        "output_final": str(args.output_final),
        "output_excluded": str(args.output_excluded),
        "n_final_rows": int(len(final_df)),
        "n_excluded_rows": int(len(excluded_df)),
        "selected_row_labels": {
            field: normalize_valence(str(selected_row.get(field, "") or "")) if field in VALENCE_FIELDS else str(selected_row.get(field, "") or "")
            for field in PRIMARY_FIELDS
            if field in workbook.columns
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
