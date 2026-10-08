import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_apertus_zero_shot import build_comparison, read_gold, read_predictions, summarize


CORE_FIELDS = [
    "relevance",
    "disease_1",
    "has_relevant_5c",
    "c5_presence_d1",
    "valence_confidence_d1",
    "valence_complacency_d1",
    "valence_constraints_d1",
    "valence_calculation_d1",
    "valence_collective_d1",
]

VALENCE_FIELDS = [
    "valence_confidence_d1",
    "valence_complacency_d1",
    "valence_constraints_d1",
    "valence_calculation_d1",
    "valence_collective_d1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a true two-stage Qwen Stage A evaluation: broad Qwen stage-1 relevance gate, "
            "then adapted Qwen stage-2 hierarchy only for rows passing the gate."
        )
    )
    parser.add_argument(
        "--gold",
        default="data/locked/gold100/gold100_test_gold_labels.csv",
        help="Current locked gold100 label CSV.",
    )
    parser.add_argument(
        "--stage1",
        action="append",
        required=True,
        help="Stage-1 relevance prediction JSONL. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--stage2",
        required=True,
        help="Forced-relevant stage-2 hierarchy prediction JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/stage_a_eval/qwen_clean_v2_gold300_readj_gated_v1",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default="gold100_qwen_stage_a_gated_v1",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def load_jsonl_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                doc_id = str(record["doc_id"])
                if doc_id in records:
                    raise ValueError(f"Duplicate doc_id {doc_id} across JSONL inputs.")
                records[doc_id] = record
    return records


def not_relevant_record(doc_id: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "doc_id": doc_id,
        "relevance": "not_relevant",
        "disease_1": None,
        "has_relevant_5c": None,
        "c5_presence_d1": [],
    }
    for field in VALENCE_FIELDS:
        record[field] = None
    return record


def build_gated_predictions(
    gold_doc_ids: list[str],
    stage1: dict[str, dict[str, Any]],
    stage2: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing_stage1 = sorted(set(gold_doc_ids) - set(stage1))
    if missing_stage1:
        raise ValueError(f"Missing stage-1 predictions for doc_ids: {missing_stage1[:20]}")

    gated: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    missing_routed_stage2: list[str] = []
    for doc_id in gold_doc_ids:
        stage1_record = stage1[doc_id]
        relevance = str(stage1_record.get("relevance", "")).strip()
        if relevance == "relevant":
            if doc_id not in stage2:
                missing_routed_stage2.append(doc_id)
                continue
            pred = {field: stage2[doc_id].get(field) for field in CORE_FIELDS}
            pred["doc_id"] = doc_id
            pred["relevance"] = "relevant"
            routed_to_stage2 = True
        elif relevance == "not_relevant":
            pred = not_relevant_record(doc_id)
            routed_to_stage2 = False
        else:
            raise ValueError(f"Unexpected stage-1 relevance for doc_id {doc_id}: {relevance!r}")
        gated.append(pred)
        audit_rows.append(
            {
                "doc_id": doc_id,
                "stage1_relevance": relevance,
                "stage1_reason_code": stage1_record.get("reason_code"),
                "routed_to_stage2": routed_to_stage2,
            }
        )
    if missing_routed_stage2:
        raise ValueError(f"Missing routed stage-2 predictions for doc_ids: {missing_routed_stage2[:20]}")
    return gated, audit_rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_df = read_gold(gold_path)
    gold_doc_ids = gold_df["doc_id"].astype(str).tolist()
    stage1 = load_jsonl_records([Path(path) for path in args.stage1])
    stage2 = load_jsonl_records([Path(args.stage2)])
    gated, audit_rows = build_gated_predictions(gold_doc_ids, stage1, stage2)

    predictions_path = output_dir / f"{args.output_prefix}_predictions.jsonl"
    audit_path = output_dir / f"{args.output_prefix}_routing_audit.csv"
    comparison_path = output_dir / f"{args.output_prefix}_comparison.csv"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"

    write_jsonl(predictions_path, gated)

    import pandas as pd

    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    pred_df = read_predictions(predictions_path)
    comparison = build_comparison(gold_df, pred_df)
    summary = summarize(comparison)
    summary["inputs"] = {
        "gold": str(gold_path),
        "stage1": [str(path) for path in args.stage1],
        "stage2": str(args.stage2),
        "gated_predictions": str(predictions_path),
    }
    summary["routing"] = {
        "n_total": len(audit_rows),
        "n_routed_to_stage2": sum(1 for row in audit_rows if row["routed_to_stage2"]),
        "n_gated_out": sum(1 for row in audit_rows if not row["routed_to_stage2"]),
    }

    comparison.to_csv(comparison_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Saved gated predictions: {predictions_path}")
    print(f"Saved routing audit: {audit_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Relevance accuracy: {summary['field_accuracy']['relevance']:.3f}")
    print(f"Exact match all fields: {summary['exact_match_all_fields']:.3f}")
    print(f"Construct presence macro F1: {summary['construct_presence_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
