from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "annotations" / "backups" / "exports" / "active_learning_project_export.json"
DEFAULT_RAW = ROOT / "data" / "processed" / "active_learning" / "active_learning_export_parsed_raw.csv"
DEFAULT_TIDY = ROOT / "data" / "processed" / "active_learning" / "active_learning_export_parsed_tidy.csv"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "active_learning" / "active_learning_export_parse_manifest.json"
GOLD100 = ROOT / "data" / "locked" / "gold100" / "gold100_test_gold_labels.csv"
GOLD300 = ROOT / "data" / "locked" / "gold300_train" / "gold300_train_adjudicated.csv"
PRIMARY_CONSTRUCTS = [
    "confidence",
    "complacency",
    "constraints",
    "calculation",
    "collective_responsibility",
]
PRIMARY_VALENCE_FIELDS = {
    "confidence": "valence_confidence_d1",
    "complacency": "valence_complacency_d1",
    "constraints": "valence_constraints_d1",
    "calculation": "valence_calculation_d1",
    "collective_responsibility": "valence_collective_d1",
}
ALL_PRIMARY_FIELDS = [
    "diseases_quick",
    "diseases_quick_other_text",
    "has_relevant_5c",
    "disease_1",
    "disease_1_other_text",
    "c5_presence_d1",
    *PRIMARY_VALENCE_FIELDS.values(),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a single-annotation active-learning Label Studio export.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-tidy", type=Path, default=DEFAULT_TIDY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_or_none(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        return value[0]
    return value


def parse_choice_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def serialize_choice_list(values: list[str]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return json.dumps(cleaned, ensure_ascii=False)


def clear_field(row: dict[str, Any], key: str, audit: Counter[str], reason: str) -> None:
    if str(row.get(key) or "").strip():
        audit[f"cleared_{reason}_{key}"] += 1
    row[key] = ""


def normalize_text_for_signature(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def text_signature(text: str) -> str:
    normalized = normalize_text_for_signature(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def clean_annotation_row(row: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    cleaned = dict(row)
    audit: Counter[str] = Counter()

    relevance = str(cleaned.get("relevance") or "").strip()
    diseases_quick_values = parse_choice_list(cleaned.get("diseases_quick"))
    presence_values = parse_choice_list(cleaned.get("c5_presence_d1"))

    cleaned["diseases_quick"] = serialize_choice_list(diseases_quick_values)
    cleaned["c5_presence_d1"] = serialize_choice_list(presence_values)

    if "other" not in diseases_quick_values:
        clear_field(cleaned, "diseases_quick_other_text", audit, "diseases_quick")
    if str(cleaned.get("disease_1") or "").strip() != "other":
        clear_field(cleaned, "disease_1_other_text", audit, "disease_1")

    if relevance != "relevant":
        audit["rows_not_relevant"] += 1
        for key in ALL_PRIMARY_FIELDS:
            clear_field(cleaned, key, audit, "not_relevant")
        cleaned["qa_hierarchy_issue"] = ""
        cleaned["qa_hierarchy_cleanup_applied"] = "yes"
        cleaned["qa_text_signature"] = text_signature(cleaned.get("text", ""))
        return cleaned, audit

    has_relevant_5c = str(cleaned.get("has_relevant_5c") or "").strip()
    cleanup_applied = False
    hierarchy_issues: list[str] = []

    if has_relevant_5c != "yes":
        for key in ["c5_presence_d1", *PRIMARY_VALENCE_FIELDS.values()]:
            before = str(cleaned.get(key) or "").strip()
            clear_field(cleaned, key, audit, "has_relevant_5c_no")
            if before:
                cleanup_applied = True
        cleaned["c5_presence_d1"] = ""
    else:
        if not presence_values:
            hierarchy_issues.append("has_relevant_5c_yes_but_no_c5_presence_d1")
            for key in PRIMARY_VALENCE_FIELDS.values():
                before = str(cleaned.get(key) or "").strip()
                clear_field(cleaned, key, audit, "missing_presence")
                if before:
                    cleanup_applied = True
        else:
            for construct, valence_field in PRIMARY_VALENCE_FIELDS.items():
                valence_value = str(cleaned.get(valence_field) or "").strip()
                if construct not in presence_values:
                    if valence_value:
                        cleanup_applied = True
                    clear_field(cleaned, valence_field, audit, f"construct_absent_{construct}")
                elif not valence_value:
                    hierarchy_issues.append(f"missing_{valence_field}")

    cleaned["qa_hierarchy_issue"] = ";".join(hierarchy_issues)
    cleaned["qa_hierarchy_cleanup_applied"] = "yes" if cleanup_applied else "no"
    cleaned["qa_text_signature"] = text_signature(cleaned.get("text", ""))
    if hierarchy_issues:
        audit["rows_with_hierarchy_issue"] += 1
        for issue in hierarchy_issues:
            audit[f"issue_{issue}"] += 1
    if cleanup_applied:
        audit["rows_with_cleanup_applied"] += 1
    return cleaned, audit


def parse_one_annotation(result_list: list[dict[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in result_list:
        key = item.get("from_name")
        value = item.get("value", {})
        if "choices" in value:
            choices = value["choices"]
            if len(choices) == 1:
                parsed[key] = choices[0]
            else:
                parsed[key] = json.dumps(choices, ensure_ascii=False)
        elif "text" in value:
            texts = value["text"]
            if len(texts) == 1:
                parsed[key] = texts[0]
            else:
                parsed[key] = json.dumps(texts, ensure_ascii=False)
        else:
            parsed[key] = None
    return parsed


def choose_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = task.get("annotations") or []
    completed = [ann for ann in annotations if not ann.get("was_cancelled", False)]
    if not completed:
        return None

    def sort_key(ann: dict[str, Any]) -> tuple[str, int]:
        updated = str(ann.get("updated_at") or ann.get("created_at") or "")
        ann_id = int(ann.get("id") or 0)
        return (updated, ann_id)

    completed.sort(key=sort_key, reverse=True)
    return completed[0]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_doc_id_set(path: Path) -> set[str]:
    import pandas as pd

    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for candidate in ["source_doc_id", "doc_id"]:
        if candidate in df.columns:
            return set(df[candidate].astype(str))
    return set()


def main() -> None:
    args = parse_args()
    tasks = load_json(args.input)

    raw_rows: list[dict[str, Any]] = []
    tidy_rows: list[dict[str, Any]] = []
    skipped_no_annotation = 0
    batch_counter: Counter[str] = Counter()
    relevance_counter: Counter[str] = Counter()
    qa_counter: Counter[str] = Counter()

    for task in tasks:
        data = task.get("data", {})
        ann = choose_annotation(task)
        if ann is None:
            skipped_no_annotation += 1
            continue

        result = ann.get("result", [])
        parsed = parse_one_annotation(result)

        ls_task_id = task.get("id")
        doc_id = str(data.get("doc_id") or "")
        source_doc_id = str(data.get("source_doc_id") or doc_id)
        al_batch = str(data.get("al_batch") or "")
        batch_counter[al_batch] += 1
        relevance_counter[str(parsed.get("relevance") or "")] += 1

        raw_rows.append(
            {
                "labelstudio_task_id": ls_task_id,
                "doc_id": doc_id,
                "source_doc_id": source_doc_id,
                "annotation_id": ann.get("id"),
                "created_at": ann.get("created_at"),
                "updated_at": ann.get("updated_at"),
                "al_batch": al_batch,
                "subset_role": data.get("subset_role", ""),
                "selection_flags": data.get("selection_flags", ""),
                "selection_summary": data.get("selection_summary", ""),
                "title": data.get("title", ""),
                "text": data.get("text", ""),
                "annotation_raw": json.dumps(result, ensure_ascii=False),
            }
        )

        row: dict[str, Any] = {
            "labelstudio_task_id": ls_task_id,
            "doc_id": doc_id,
            "source_doc_id": source_doc_id,
            "annotation_id": ann.get("id"),
            "created_at": ann.get("created_at"),
            "updated_at": ann.get("updated_at"),
            "al_batch": al_batch,
            "subset_role": data.get("subset_role", ""),
            "selection_flags": data.get("selection_flags", ""),
            "selection_summary": data.get("selection_summary", ""),
            "title": data.get("title", ""),
            "text": data.get("text", ""),
        }
        row.update(parsed)
        cleaned_row, audit = clean_annotation_row(row)
        tidy_rows.append(cleaned_row)
        qa_counter.update(audit)

    write_csv(raw_rows, args.output_raw)
    write_csv(tidy_rows, args.output_tidy)

    gold100_ids = load_doc_id_set(GOLD100)
    gold300_ids = load_doc_id_set(GOLD300)
    exported_source_ids = {str(row["source_doc_id"]) for row in tidy_rows}
    overlap_gold100 = sorted(exported_source_ids & gold100_ids)
    overlap_gold300 = sorted(exported_source_ids & gold300_ids)

    manifest = {
        "input": str(args.input),
        "n_tasks_total": len(tasks),
        "n_annotations_parsed": len(tidy_rows),
        "n_skipped_no_annotation": skipped_no_annotation,
        "output_raw": str(args.output_raw),
        "output_tidy": str(args.output_tidy),
        "batch_counts": {k: int(v) for k, v in batch_counter.items()},
        "relevance_counts": {k: int(v) for k, v in relevance_counter.items()},
        "qa_counts": {k: int(v) for k, v in qa_counter.items()},
        "overlap_checks": {
            "gold100_overlap_count": len(overlap_gold100),
            "gold300_overlap_count": len(overlap_gold300),
            "gold100_overlap_examples": overlap_gold100[:10],
            "gold300_overlap_examples": overlap_gold300[:10],
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
