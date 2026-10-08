import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_stage_a_gated_qwen_eval import build_gated_predictions, load_jsonl_records, write_jsonl
from evaluate_apertus_zero_shot import build_comparison, read_gold, read_predictions, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate a local two-stage gate/extractor matrix from existing "
            "stage-1 relevance JSONL files and forced-relevant stage-2 JSONL files."
        )
    )
    parser.add_argument("--gold", default="data/locked/gold100/gold100_test_gold_labels.csv")
    parser.add_argument(
        "--gate",
        action="append",
        required=True,
        help=(
            "Gate spec as ID=path[,path2]. Multiple paths are allowed for chunked or "
            "repair runs with disjoint doc_ids."
        ),
    )
    parser.add_argument(
        "--extractor",
        action="append",
        required=True,
        help=(
            "Extractor spec as ID=path[,repair_path]. Multiple paths are allowed "
            "for disjoint repair runs."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_named_paths(raw: str) -> tuple[str, list[Path]]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=path spec, got: {raw}")
    name, paths_raw = raw.split("=", 1)
    name = name.strip()
    paths = [Path(part.strip()) for part in paths_raw.split(",") if part.strip()]
    if not name or not paths:
        raise ValueError(f"Invalid named path spec: {raw}")
    return name, paths


def load_gates(specs: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    gates: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        name, paths = parse_named_paths(spec)
        gates[name] = load_jsonl_records(paths)
    return gates


def load_extractors(specs: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    extractors: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        name, paths = parse_named_paths(spec)
        extractors[name] = load_jsonl_records(paths)
    return extractors


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_path = Path(args.gold)
    gold_df = read_gold(gold_path)
    gold_doc_ids = gold_df["doc_id"].astype(str).tolist()
    gates = load_gates(args.gate)
    extractors = load_extractors(args.extractor)

    overview_rows: list[dict[str, Any]] = []

    for gate_id, gate_records in gates.items():
        for extractor_id, extractor_records in extractors.items():
            combo_id = f"{gate_id}_{extractor_id}"
            combo_dir = output_dir / combo_id
            combo_dir.mkdir(parents=True, exist_ok=True)

            try:
                gated, audit_rows = build_gated_predictions(gold_doc_ids, gate_records, extractor_records)
                predictions_path = combo_dir / f"{combo_id}_gated_predictions.jsonl"
                audit_path = combo_dir / f"{combo_id}_routing_audit.csv"
                comparison_path = combo_dir / f"{combo_id}_comparison.csv"
                summary_path = combo_dir / f"{combo_id}_summary.json"

                write_jsonl(predictions_path, gated)
                pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
                pred_df = read_predictions(predictions_path)
                comparison = build_comparison(gold_df, pred_df)
                summary = summarize(comparison)
                summary["inputs"] = {
                    "gold": str(gold_path),
                    "gate_id": gate_id,
                    "extractor_id": extractor_id,
                    "gated_predictions": str(predictions_path),
                }
                summary["routing"] = {
                    "n_total": len(audit_rows),
                    "n_routed_to_stage2": sum(1 for row in audit_rows if row["routed_to_stage2"]),
                    "n_gated_out": sum(1 for row in audit_rows if not row["routed_to_stage2"]),
                }
                comparison.to_csv(comparison_path, index=False)
                write_summary(summary_path, summary)
                overview_rows.append(
                    {
                        "combo_id": combo_id,
                        "gate_id": gate_id,
                        "extractor_id": extractor_id,
                        "status": "ok",
                        "summary_json": str(summary_path),
                        "relevance_accuracy": summary["field_accuracy"].get("relevance"),
                        "has_relevant_5c_accuracy": summary["field_accuracy"].get("has_relevant_5c"),
                        "construct_presence_macro_f1": summary.get("construct_presence_macro_f1"),
                        "construct_presence_micro_f1": (summary.get("construct_presence_micro") or {}).get("f1"),
                        "exact_match_all_fields_internal": summary.get("exact_match_all_fields"),
                        "n_routed_to_stage2": summary["routing"]["n_routed_to_stage2"],
                        "n_gated_out": summary["routing"]["n_gated_out"],
                    }
                )
                print(f"{combo_id}: ok")
            except Exception as exc:  # noqa: BLE001
                overview_rows.append(
                    {
                        "combo_id": combo_id,
                        "gate_id": gate_id,
                        "extractor_id": extractor_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"{combo_id}: error: {exc}")

    overview_path = output_dir / "gated_matrix_overview.csv"
    fields = sorted({key for row in overview_rows for key in row})
    with overview_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overview_rows)
    print(f"Saved overview: {overview_path}")


if __name__ == "__main__":
    main()
