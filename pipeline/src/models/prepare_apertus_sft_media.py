import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.media_text import build_model_text

DEFAULT_INPUT = ROOT / "data" / "processed" / "gold300_train_adjudicated.csv"
DEFAULT_SPLIT = ROOT / "data" / "processed" / "gold400_split_assignment.csv"
DEFAULT_DISAGREEMENTS = ROOT / "data" / "processed" / "disagreements_train_after_adjudication.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "apertus_sft_media"
DEFAULT_SCHEMA = ROOT / "configs" / "apertus_5c_zero_shot_schema.json"
DEFAULT_PROMPT = ROOT / "configs" / "apertus_5c_zero_shot_prompt.md"
DEFAULT_SYSTEM_PROMPT = ROOT / "configs" / "apertus_5c_sft_system_prompt_v7_broad_relevance.txt"

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
ALLOWED_RELEVANCE = {"relevant", "not_relevant"}
ALLOWED_DISEASES = {"covid19", "influenza", "measles", "hpv", "unspecified", "other"}
ALLOWED_HAS_5C = {"yes", "no"}
ALLOWED_VALENCES = {-1, 0, 1}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Apertus supervised fine-tuning JSONL from the adjudicated train split."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Adjudicated train CSV.")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT), help="Gold split assignment CSV.")
    parser.add_argument("--disagreements", default=str(DEFAULT_DISAGREEMENTS), help="Train disagreement CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Schema recorded in manifest.")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="Prompt recorded in manifest.")
    parser.add_argument(
        "--system-prompt",
        default=str(DEFAULT_SYSTEM_PROMPT),
        help="System prompt file used in SFT messages and recorded in the manifest.",
    )
    parser.add_argument("--dev-fraction", type=float, default=0.1, help="Train-side dev fraction.")
    parser.add_argument("--seed", type=int, default=20260417, help="Deterministic split seed.")
    parser.add_argument(
        "--allow-unresolved-preview",
        action="store_true",
        help="Allow a preview dataset by excluding doc_ids with unresolved train disagreements.",
    )
    parser.add_argument(
        "--clean-obvious-artifacts",
        action="store_true",
        help="Apply conservative OCR/layout artifact cleanup when building model text.",
    )
    return parser.parse_args()


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_presence(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    text = str(value).strip()
    if text in {"[]", "null", "None"}:
        return []
    if text.startswith("["):
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            values = ast.literal_eval(text)
    else:
        values = [text]
    values = [str(item) for item in values if str(item).strip()]
    invalid = [item for item in values if item not in CONSTRUCTS]
    if invalid:
        raise ValueError(f"Invalid construct label(s): {invalid}")
    return sorted(set(values), key=CONSTRUCTS.index)


def normalize_valence(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.startswith("'"):
        text = text[1:]
    mapping = {
        "-1": -1,
        "-1.0": -1,
        "0": 0,
        "0.0": 0,
        "1": 1,
        "+1": 1,
        "1.0": 1,
        "+1.0": 1,
    }
    if text not in mapping:
        raise ValueError(f"Invalid valence value: {value}")
    return mapping[text]


def validate_train_split(df: pd.DataFrame, split_df: pd.DataFrame) -> None:
    if len(df) != df["doc_id"].nunique():
        raise ValueError("Training input has duplicate doc_id values.")
    merged = df[["doc_id"]].merge(split_df[["doc_id", "split"]], on="doc_id", how="left")
    bad = merged[merged["split"] != "train"]
    if not bad.empty:
        raise ValueError(f"Input contains rows outside train split: {bad['doc_id'].head(10).tolist()}")


def unresolved_doc_ids(disagreement_path: str | Path) -> set[str]:
    path = Path(disagreement_path)
    if not path.exists():
        return set()
    disagreements = read_csv(path)
    if disagreements.empty:
        return set()
    return set(disagreements["doc_id"].astype(str))


def build_label_record(row: dict[str, Any]) -> dict[str, Any]:
    doc_id = row["doc_id"]
    relevance = row["ann1_relevance"]
    disease_1 = row["ann1_disease_1"] or None
    has_5c = row["ann1_has_relevant_5c"] or None
    presence = parse_presence(row["ann1_c5_presence_d1"])

    if relevance not in ALLOWED_RELEVANCE:
        raise ValueError(f"Invalid relevance for doc_id {doc_id}: {relevance}")

    labels = {
        "relevance": relevance,
        "disease_1": disease_1,
        "has_relevant_5c": has_5c,
        "c5_presence_d1": presence,
    }

    if relevance == "not_relevant":
        labels["disease_1"] = None
        labels["has_relevant_5c"] = None
        labels["c5_presence_d1"] = []
        for field in VALENCE_FIELDS.values():
            labels[field] = None
        return labels

    if disease_1 not in ALLOWED_DISEASES:
        raise ValueError(f"Invalid disease_1 for doc_id {doc_id}: {disease_1}")
    if has_5c not in ALLOWED_HAS_5C:
        raise ValueError(f"Invalid has_relevant_5c for doc_id {doc_id}: {has_5c}")

    if has_5c == "no":
        labels["c5_presence_d1"] = []
        for field in VALENCE_FIELDS.values():
            labels[field] = None
        return labels

    if not presence:
        raise ValueError(f"doc_id {doc_id} has has_relevant_5c=yes but no c5_presence_d1")

    for construct, field in VALENCE_FIELDS.items():
        value = normalize_valence(row[f"ann1_{field}"])
        if construct in presence:
            if value not in ALLOWED_VALENCES:
                raise ValueError(f"Missing or invalid {field} for doc_id {doc_id}")
            labels[field] = value
        else:
            labels[field] = None
    return labels


def split_key(doc_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).hexdigest()


def assign_train_dev(df: pd.DataFrame, dev_fraction: float, seed: int) -> pd.DataFrame:
    if not 0 < dev_fraction < 0.5:
        raise ValueError("--dev-fraction must be greater than 0 and less than 0.5")
    ordered = df.copy()
    ordered["_split_key"] = ordered["doc_id"].astype(str).apply(lambda value: split_key(value, seed))
    ordered = ordered.sort_values("_split_key").reset_index(drop=True)
    n_dev = max(1, round(len(ordered) * dev_fraction))
    ordered["sft_split"] = "train"
    ordered.loc[: n_dev - 1, "sft_split"] = "dev"
    return ordered.drop(columns=["_split_key"])


def build_training_record(
    row: dict[str, Any],
    system_prompt: str,
    clean_obvious_artifacts: bool = False,
) -> dict[str, Any]:
    labels = build_label_record(row)
    assistant = json.dumps(labels, ensure_ascii=False, sort_keys=True)
    model_text = build_model_text(
        str(row.get("title", "")),
        str(row["text"]),
        clean_obvious_artifacts=clean_obvious_artifacts,
    )
    return {
        "doc_id": row["doc_id"],
        "labelstudio_task_id": row["labelstudio_task_id"],
        "source_doc_id": row["source_doc_id"],
        "source_tier": "mainstream_media",
        "sft_split": row["sft_split"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Article:\n{model_text}"},
            {"role": "assistant", "content": assistant},
        ],
        "labels": labels,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def count_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [record["labels"] for record in records]
    df = pd.DataFrame(rows)
    return {
        "relevance": df["relevance"].value_counts(dropna=False).to_dict(),
        "disease_1": df["disease_1"].fillna("null").value_counts(dropna=False).to_dict(),
        "has_relevant_5c": df["has_relevant_5c"].fillna("null").value_counts(dropna=False).to_dict(),
        "construct_presence": {
            construct: int(df["c5_presence_d1"].apply(lambda values, c=construct: c in values).sum())
            for construct in CONSTRUCTS
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    df = read_csv(input_path)
    split_df = read_csv(args.split)
    validate_train_split(df, split_df)
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()

    unresolved = unresolved_doc_ids(args.disagreements)
    preview_mode = False
    if unresolved:
        if not args.allow_unresolved_preview:
            raise ValueError(
                f"Training split still has unresolved disagreements for {len(unresolved)} doc_id(s). "
                "Finish train adjudication first, or rerun with --allow-unresolved-preview to create "
                "a non-final preview that excludes unresolved doc_ids."
            )
        preview_mode = True
        df = df[~df["doc_id"].astype(str).isin(unresolved)].copy()

    df = assign_train_dev(df, args.dev_fraction, args.seed)
    records = []
    excluded_invalid = []
    for row in df.to_dict(orient="records"):
        try:
            records.append(
                build_training_record(
                    row,
                    system_prompt,
                    clean_obvious_artifacts=args.clean_obvious_artifacts,
                )
            )
        except ValueError as exc:
            if not preview_mode:
                raise
            excluded_invalid.append({"doc_id": row["doc_id"], "reason": str(exc)})

    if not records:
        raise ValueError("No valid records available for SFT dataset creation.")

    train_records = [record for record in records if record["sft_split"] == "train"]
    dev_records = [record for record in records if record["sft_split"] == "dev"]

    train_path = output_dir / "media_sft_train.jsonl"
    dev_path = output_dir / "media_sft_dev.jsonl"
    all_path = output_dir / "media_sft_all.jsonl"
    manifest_path = output_dir / "media_sft_manifest.json"

    write_jsonl(train_path, train_records)
    write_jsonl(dev_path, dev_records)
    write_jsonl(all_path, records)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "split_assignment": str(Path(args.split)),
        "disagreement_check": str(Path(args.disagreements)),
        "preview_mode": preview_mode,
        "training_ready": not preview_mode and not excluded_invalid,
        "excluded_unresolved_doc_ids": sorted(unresolved) if preview_mode else [],
        "excluded_invalid_records": excluded_invalid,
        "n_input_rows": int(len(read_csv(input_path))),
        "n_records": int(len(records)),
        "n_train": int(len(train_records)),
        "n_dev": int(len(dev_records)),
        "dev_fraction": args.dev_fraction,
        "seed": args.seed,
        "outputs": {
            "train_jsonl": str(train_path),
            "dev_jsonl": str(dev_path),
            "all_jsonl": str(all_path),
        },
        "sha256": {
            "input": sha256_file(input_path),
            "schema": sha256_file(args.schema),
            "prompt": sha256_file(args.prompt),
            "system_prompt": sha256_file(args.system_prompt),
            "train_jsonl": sha256_file(train_path),
            "dev_jsonl": sha256_file(dev_path),
            "all_jsonl": sha256_file(all_path),
        },
        "system_prompt": str(Path(args.system_prompt)),
        "label_counts": count_labels(records),
        "policy": {
            "source_tier": "mainstream_media",
            "locked_test_excluded": True,
            "train_side_dev_only": True,
            "unresolved_train_disagreements_block_final_training": True,
            "clean_obvious_artifacts": bool(args.clean_obvious_artifacts),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    mode = "PREVIEW, not training-ready" if preview_mode else "FINAL, training-ready"
    print(f"Saved SFT train JSONL: {train_path}")
    print(f"Saved SFT dev JSONL: {dev_path}")
    print(f"Saved SFT manifest: {manifest_path}")
    print(f"Mode: {mode}")
    print(f"Records: total={len(records)}, train={len(train_records)}, dev={len(dev_records)}")


if __name__ == "__main__":
    main()
