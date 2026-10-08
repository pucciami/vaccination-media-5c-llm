from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "active_learning" / "active_learning_export_parsed_tidy.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "active_learning" / "active_learning_merge_manifest.json"
DEFAULT_REVIEW = ROOT / "data" / "processed" / "active_learning" / "active_learning_merge_ready.csv"
GOLD100_REF = ROOT / "data" / "locked" / "gold100" / "gold100_test_locked_reference.csv"
GOLD300 = ROOT / "data" / "locked" / "gold300_train" / "gold300_train_adjudicated.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a train/dev merge manifest from parsed AL annotations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-review", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def parse_presence(value: str) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def text_signature(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def training_dedup_signature(title: str, text: str, prefix_chars: int = 4000) -> str:
    normalized_title = normalize_text(title)
    normalized_text = normalize_text(text)
    payload = f"{normalized_title}||{normalized_text[:prefix_chars]}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def label_signature(row: pd.Series) -> str:
    keys = [
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
    payload = {key: str(row.get(key, "") or "") for key in keys}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_text_signature_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" not in df.columns:
        return set()
    return {text_signature(value) for value in df["text"].fillna("")}


def load_source_doc_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "source_doc_id" not in df.columns:
        return set()
    return set(df["source_doc_id"].fillna("").astype(str))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if df.empty:
        manifest = {
            "input": str(args.input),
            "n_rows": 0,
            "merge_rule": "train/dev only; never merge into locked gold100",
        }
        args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.output_review:
            df.to_csv(args.output_review, index=False, encoding="utf-8-sig")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    df["calc_present"] = df["c5_presence_d1"].apply(parse_presence).apply(lambda vals: "calculation" in vals)
    df["qa_text_signature"] = df["text"].apply(text_signature)
    df["qa_training_dedup_signature"] = df.apply(
        lambda row: training_dedup_signature(row.get("title", ""), row.get("text", "")),
        axis=1,
    )
    df["qa_label_signature"] = df.apply(label_signature, axis=1)

    sort_columns = [column for column in ["updated_at", "created_at", "annotation_id", "labelstudio_task_id"] if column in df.columns]
    df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    df["training_dedup_group_size"] = df.groupby("qa_training_dedup_signature")["doc_id"].transform("size").astype(int)
    df["training_dedup_rank"] = df.groupby("qa_training_dedup_signature").cumcount() + 1
    df["training_dedup_cluster_label_nunique"] = (
        df.groupby("qa_training_dedup_signature")["qa_label_signature"].transform("nunique").astype(int)
    )
    df["training_dedup_disagreement"] = (
        df["training_dedup_cluster_label_nunique"] > 1
    ).map({True: "yes", False: "no"})
    df["keep_for_training_dedup_cluster"] = (
        df["training_dedup_rank"] == 1
    ).map({True: "yes", False: "no"})

    gold100_text_signatures = load_text_signature_set(GOLD100_REF)
    gold300_text_signatures = load_text_signature_set(GOLD300)
    gold100_source_ids = load_source_doc_id_set(GOLD100_REF)
    gold300_source_ids = load_source_doc_id_set(GOLD300)

    df["overlap_gold100_source_doc_id"] = df["source_doc_id"].isin(gold100_source_ids).map({True: "yes", False: "no"})
    df["overlap_gold300_source_doc_id"] = df["source_doc_id"].isin(gold300_source_ids).map({True: "yes", False: "no"})
    df["overlap_gold100_exact_text"] = df["qa_text_signature"].isin(gold100_text_signatures).map({True: "yes", False: "no"})
    df["overlap_gold300_exact_text"] = df["qa_text_signature"].isin(gold300_text_signatures).map({True: "yes", False: "no"})

    exclusion_reasons: list[str] = []
    recommended_keep: list[str] = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        if row["training_dedup_disagreement"] == "yes":
            reasons.append("duplicate_cluster_label_disagreement_requires_review")
        elif row["keep_for_training_dedup_cluster"] != "yes":
            reasons.append("duplicate_within_al_training_dedup_cluster")
        if row["overlap_gold300_source_doc_id"] == "yes":
            reasons.append("source_doc_id_overlap_gold300")
        if row["overlap_gold300_exact_text"] == "yes":
            reasons.append("exact_text_overlap_gold300")
        if row["overlap_gold100_source_doc_id"] == "yes":
            reasons.append("source_doc_id_overlap_gold100")
        if row["overlap_gold100_exact_text"] == "yes":
            reasons.append("exact_text_overlap_gold100")
        exclusion_reasons.append(";".join(reasons))
        recommended_keep.append("yes" if not reasons else "no")

    df["merge_exclusion_reason"] = exclusion_reasons
    df["recommended_keep_for_train_dev_merge"] = recommended_keep
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_review, index=False, encoding="utf-8-sig")

    duplicate_rows = df[df["training_dedup_group_size"] > 1].copy()
    duplicate_cluster_examples = (
        duplicate_rows[
            [
                "doc_id",
                "source_doc_id",
                "training_dedup_group_size",
                "training_dedup_rank",
                "training_dedup_disagreement",
                "recommended_keep_for_train_dev_merge",
            ]
        ]
        .head(20)
        .to_dict(orient="records")
    )

    manifest = {
        "input": str(args.input),
        "n_rows": int(len(df)),
        "output_review": str(args.output_review),
        "batch_counts": {str(k): int(v) for k, v in df["al_batch"].value_counts().to_dict().items()},
        "relevance_counts": {str(k): int(v) for k, v in df["relevance"].value_counts().to_dict().items()},
        "has_relevant_5c_counts": {str(k): int(v) for k, v in df["has_relevant_5c"].value_counts().to_dict().items()},
        "disease_1_counts": {str(k): int(v) for k, v in df["disease_1"].value_counts().to_dict().items()},
        "calculation_presence_count": int(df["calc_present"].sum()),
        "qa_cleanup_counts": {
            str(k): int(v) for k, v in df["qa_hierarchy_cleanup_applied"].value_counts().to_dict().items()
        },
        "qa_issue_counts": {str(k): int(v) for k, v in df["qa_hierarchy_issue"].value_counts().to_dict().items()},
        "duplicate_audit": {
            "rows_in_training_dedup_clusters": int((df["training_dedup_group_size"] > 1).sum()),
            "training_dedup_cluster_count": int((df["training_dedup_rank"] == 1).mul(df["training_dedup_group_size"] > 1).sum()),
            "duplicate_rows_dropped_for_training": int((df["keep_for_training_dedup_cluster"] == "no").sum()),
            "training_dedup_clusters_with_label_disagreement": int(
                ((df["training_dedup_rank"] == 1) & (df["training_dedup_disagreement"] == "yes")).sum()
            ),
            "method": "normalized title plus first 4000 normalized text characters",
            "examples": duplicate_cluster_examples,
        },
        "overlap_audit": {
            "gold100_source_doc_id_overlap_count": int((df["overlap_gold100_source_doc_id"] == "yes").sum()),
            "gold300_source_doc_id_overlap_count": int((df["overlap_gold300_source_doc_id"] == "yes").sum()),
            "gold100_exact_text_overlap_count": int((df["overlap_gold100_exact_text"] == "yes").sum()),
            "gold300_exact_text_overlap_count": int((df["overlap_gold300_exact_text"] == "yes").sum()),
        },
        "recommended_keep_for_train_dev_merge_count": int((df["recommended_keep_for_train_dev_merge"] == "yes").sum()),
        "selection_flag_examples": df[["doc_id", "al_batch", "selection_flags", "selection_summary"]]
        .head(20)
        .to_dict(orient="records"),
        "merge_rule": "train/dev only; never merge into locked gold100; exclude exact-text duplicates and any gold-train/test overlaps from the training merge",
        "next_step": "review duplicate/overlap flags in active_learning_merge_ready.csv, then merge the recommended keep set into the train/dev enrichment package",
    }

    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
