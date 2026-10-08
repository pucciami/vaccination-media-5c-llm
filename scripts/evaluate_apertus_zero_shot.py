import argparse
import ast
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GOLD_PATH = Path("data/locked/gold100/gold100_test_gold_labels.csv")
DEFAULT_PREDICTIONS_PATH = Path("data/processed/apertus_zero_shot/gold100_test_apertus_predictions.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/apertus_zero_shot_eval")

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

CONSTRUCTS = [
    "confidence",
    "complacency",
    "constraints",
    "calculation",
    "collective_responsibility",
]

VALENCE_FIELDS = {
    "confidence": "valence_confidence_d1",
    "complacency": "valence_complacency_d1",
    "constraints": "valence_constraints_d1",
    "calculation": "valence_calculation_d1",
    "collective_responsibility": "valence_collective_d1",
}

MULTICLASS_FIELDS = [
    "relevance",
    "disease_1",
    "has_relevant_5c",
    *VALENCE_FIELDS.values(),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate locked Apertus zero-shot predictions against adjudicated gold100 labels. "
            "Run only after inference is complete; do not pass gold labels to inference."
        )
    )
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH), help="Gold labels CSV.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH), help="Predictions JSONL or CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for evaluation outputs.")
    parser.add_argument(
        "--output-prefix",
        default="gold100_apertus_zero_shot",
        help="Filename prefix for summary/comparison/confusion outputs.",
    )
    return parser.parse_args()


def parse_presence(value: Any) -> list[str]:
    if isinstance(value, list):
        return sorted([str(item) for item in value if str(item).strip()], key=CONSTRUCTS.index)
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if text in {"", "null", "None", "nan"}:
        return []
    if text.startswith("["):
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            values = ast.literal_eval(text)
    else:
        values = [text]
    return sorted([str(item) for item in values if str(item).strip()], key=CONSTRUCTS.index)


def normalize_scalar(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "null", "None", "nan"}:
        return None
    if text in {"-1.0", "0.0", "1.0"}:
        return text[:-2]
    if text == "+1":
        return "1"
    if text == "+1.0":
        return "1"
    return text


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {"doc_id": str(record["doc_id"])}
    for field in CORE_FIELDS:
        if field == "c5_presence_d1":
            normalized[field] = parse_presence(record.get(field))
        else:
            normalized[field] = normalize_scalar(record.get(field))
    return normalized


def read_gold(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    records = [normalize_record(record) for record in df.to_dict(orient="records")]
    return pd.DataFrame(records)


def read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        records = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict(orient="records")
    normalized = [normalize_record(record) for record in records]
    return pd.DataFrame(normalized)


def binary_metrics(gold: list[bool], pred: list[bool]) -> dict[str, float | int | None]:
    tp = sum(g and p for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1_denominator = (2 * tp) + fp + fn
    f1 = (2 * tp / f1_denominator) if f1_denominator else 0.0
    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    }


def one_vs_rest_metrics(gold: list[str | None], pred: list[str | None], labels: list[str]) -> dict[str, Any]:
    per_label = {}
    for label in labels:
        per_label[label] = binary_metrics(
            [value == label for value in gold],
            [value == label for value in pred],
        )
    f1_values = [metrics["f1"] for metrics in per_label.values() if metrics["f1"] is not None]
    return {
        "labels": per_label,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
    }


def accuracy(gold: list[Any], pred: list[Any]) -> float | None:
    if not gold:
        return None
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def confusion_rows(frame: pd.DataFrame, field: str) -> list[dict[str, Any]]:
    counts = Counter(
        zip(
            [metric_scalar(value) for value in frame[f"gold_{field}"].tolist()],
            [metric_scalar(value) for value in frame[f"pred_{field}"].tolist()],
        )
    )
    return [
        {"field": field, "gold": gold, "pred": pred, "n": n}
        for (gold, pred), n in sorted(counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
    ]


def metric_scalar(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def build_comparison(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if gold_df["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id values in gold labels.")
    if pred_df["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id values in predictions.")

    merged = gold_df.merge(pred_df, on="doc_id", how="outer", suffixes=("_gold", "_pred"), indicator=True)
    missing = merged[merged["_merge"] != "both"]
    if not missing.empty:
        examples = missing[["doc_id", "_merge"]].head(10).to_dict(orient="records")
        raise ValueError(f"Gold/prediction doc_id mismatch: {examples}")

    rows = []
    for record in merged.to_dict(orient="records"):
        row = {"doc_id": record["doc_id"]}
        all_match = True
        for field in CORE_FIELDS:
            gold_value = record[f"{field}_gold"]
            pred_value = record[f"{field}_pred"]
            if field == "c5_presence_d1":
                gold_value = "|".join(gold_value)
                pred_value = "|".join(pred_value)
            else:
                gold_value = None if pd.isna(gold_value) else gold_value
                pred_value = None if pd.isna(pred_value) else pred_value
            match = gold_value == pred_value
            row[f"gold_{field}"] = gold_value
            row[f"pred_{field}"] = pred_value
            row[f"match_{field}"] = match
            all_match = all_match and match
        row["exact_match_all_fields"] = all_match
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(comparison: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": int(len(comparison)),
        "exact_match_all_fields": float(comparison["exact_match_all_fields"].mean()),
        "field_accuracy": {},
        "classification_metrics": {},
        "construct_presence": {},
        "construct_valence": {},
    }

    for field in CORE_FIELDS:
        summary["field_accuracy"][field] = float(comparison[f"match_{field}"].mean())

    for field in MULTICLASS_FIELDS:
        gold = [metric_scalar(value) for value in comparison[f"gold_{field}"].tolist()]
        pred = [metric_scalar(value) for value in comparison[f"pred_{field}"].tolist()]
        labels = sorted({value for value in gold + pred if value is not None})
        summary["classification_metrics"][field] = {
            "accuracy": accuracy(gold, pred),
            "one_vs_rest": one_vs_rest_metrics(gold, pred, labels),
        }

    summary["classification_metrics"]["relevance"]["binary_relevant"] = binary_metrics(
        [value == "relevant" for value in comparison["gold_relevance"].tolist()],
        [value == "relevant" for value in comparison["pred_relevance"].tolist()],
    )

    relevant_rows = comparison[comparison["gold_relevance"] == "relevant"]
    if not relevant_rows.empty:
        summary["classification_metrics"]["disease_1"]["accuracy_gold_relevant_only"] = accuracy(
            [metric_scalar(value) for value in relevant_rows["gold_disease_1"].tolist()],
            [metric_scalar(value) for value in relevant_rows["pred_disease_1"].tolist()],
        )
        summary["classification_metrics"]["has_relevant_5c"]["accuracy_gold_relevant_only"] = accuracy(
            [metric_scalar(value) for value in relevant_rows["gold_has_relevant_5c"].tolist()],
            [metric_scalar(value) for value in relevant_rows["pred_has_relevant_5c"].tolist()],
        )

    for construct in CONSTRUCTS:
        field = VALENCE_FIELDS[construct]
        gold_presence = [
            construct in str(value).split("|") if value else False
            for value in comparison["gold_c5_presence_d1"].tolist()
        ]
        pred_presence = [
            construct in str(value).split("|") if value else False
            for value in comparison["pred_c5_presence_d1"].tolist()
        ]
        presence_metrics = binary_metrics(gold_presence, pred_presence)
        summary["construct_presence"][construct] = presence_metrics

        gold_or_pred_present = [g or p for g, p in zip(gold_presence, pred_presence)]
        union_rows = comparison[gold_or_pred_present]
        gold_present_rows = comparison[gold_presence]
        summary["construct_valence"][construct] = {
            "field": field,
            "accuracy_when_gold_or_pred_present": accuracy(
                [metric_scalar(value) for value in union_rows[f"gold_{field}"].tolist()],
                [metric_scalar(value) for value in union_rows[f"pred_{field}"].tolist()],
            ),
            "accuracy_when_gold_present": accuracy(
                [metric_scalar(value) for value in gold_present_rows[f"gold_{field}"].tolist()],
                [metric_scalar(value) for value in gold_present_rows[f"pred_{field}"].tolist()],
            ),
            "n_gold_or_pred_present": int(len(union_rows)),
            "n_gold_present": int(len(gold_present_rows)),
        }

    construct_f1 = [
        metrics["f1"] for metrics in summary["construct_presence"].values() if metrics["f1"] is not None
    ]
    summary["construct_presence_macro_f1"] = sum(construct_f1) / len(construct_f1) if construct_f1 else None

    micro_gold = []
    micro_pred = []
    for construct in CONSTRUCTS:
        micro_gold.extend(
            [
                construct in str(value).split("|") if value else False
                for value in comparison["gold_c5_presence_d1"].tolist()
            ]
        )
        micro_pred.extend(
            [
                construct in str(value).split("|") if value else False
                for value in comparison["pred_c5_presence_d1"].tolist()
            ]
        )
    summary["construct_presence_micro"] = binary_metrics(micro_gold, micro_pred)

    return summary


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_df = read_gold(gold_path)
    pred_df = read_predictions(predictions_path)
    comparison = build_comparison(gold_df, pred_df)
    summary = summarize(comparison)
    summary["inputs"] = {
        "gold": str(gold_path),
        "predictions": str(predictions_path),
    }

    comparison_path = output_dir / f"{args.output_prefix}_comparison.csv"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    confusion_path = output_dir / f"{args.output_prefix}_confusions.csv"

    comparison.to_csv(comparison_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    confusion = []
    for field in MULTICLASS_FIELDS:
        confusion.extend(confusion_rows(comparison, field))
    pd.DataFrame(confusion).to_csv(confusion_path, index=False)

    print(f"Saved comparison CSV: {comparison_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved confusion CSV: {confusion_path}")
    print(f"Exact match all fields: {summary['exact_match_all_fields']:.3f}")
    print(f"Relevance accuracy: {summary['field_accuracy']['relevance']:.3f}")
    if summary["construct_presence_macro_f1"] is not None:
        print(f"Construct presence macro F1: {summary['construct_presence_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
