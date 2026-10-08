from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD300 = ROOT / "data" / "locked" / "gold300_train" / "gold300_train_adjudicated.csv"
DEFAULT_CLEAN_V2 = ROOT / "data" / "processed" / "clean_v2" / "documents_media_clean_v2.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "clean_v2_gold300_relevance_readjudicated_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the 300-article training package rebased onto clean-v2 article text."
    )
    parser.add_argument("--gold300", type=Path, default=DEFAULT_GOLD300)
    parser.add_argument("--clean-v2", type=Path, default=DEFAULT_CLEAN_V2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_labels(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    presence = df["ann1_c5_presence_d1"].fillna("").astype(str)
    constructs = ["confidence", "complacency", "constraints", "calculation", "collective_responsibility"]
    return {
        "relevance": df["ann1_relevance"].fillna("").value_counts(dropna=False).to_dict(),
        "disease_1": df["ann1_disease_1"].fillna("null").replace("", "null").value_counts(dropna=False).to_dict(),
        "has_relevant_5c": df["ann1_has_relevant_5c"].fillna("null").replace("", "null").value_counts(dropna=False).to_dict(),
        "construct_presence": {
            construct: int(presence.str.contains(construct, regex=False).sum())
            for construct in constructs
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(args.gold300, dtype=str, keep_default_na=False)
    clean = pd.read_parquet(args.clean_v2).copy()
    clean["doc_id"] = clean["doc_id"].astype(str)
    clean_lookup = clean.set_index("doc_id", drop=False)

    if len(gold) != 300:
        raise ValueError(f"Expected 300 gold train rows, found {len(gold)}")
    if len(gold) != gold["doc_id"].astype(str).nunique():
        raise ValueError("Gold300 has duplicate doc_id values.")

    missing = sorted(set(gold["source_doc_id"].astype(str)) - set(clean_lookup.index))
    if missing:
        raise ValueError(f"Missing clean-v2 source rows for {len(missing)} gold300 records.")

    out = gold.copy()
    original_title = out.get("title", pd.Series([""] * len(out))).astype(str)
    original_text = out["text"].astype(str)
    clean_rows = clean_lookup.loc[out["source_doc_id"].astype(str)]

    for column in ["title", "text", "published_at", "outlet"]:
        if column in clean_rows.columns:
            out[column] = clean_rows[column].astype(str).to_list()
    out["package_origin"] = "gold300_locked"
    out["clean_v2_changed"] = clean_rows["clean_v2_changed"].astype(bool).to_list()
    out["clean_v2_removed_chars"] = clean_rows["clean_v2_removed_chars"].astype(int).to_list()

    output_csv = args.output_dir / "media_train_clean_v2_gold300.csv"
    manifest_path = args.output_dir / "media_train_clean_v2_gold300_manifest.json"
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold300_input": str(args.gold300),
        "clean_v2_input": str(args.clean_v2),
        "outputs": {"train_package": str(output_csv)},
        "counts": {"gold300_rows": int(len(out))},
        "label_counts": count_labels(out),
        "clean_v2_audit": {
            "title_changed_count": int((original_title.reset_index(drop=True) != out["title"].astype(str).reset_index(drop=True)).sum()),
            "text_changed_count": int((original_text.reset_index(drop=True) != out["text"].astype(str).reset_index(drop=True)).sum()),
            "clean_v2_changed_source_count": int(pd.Series(out["clean_v2_changed"]).astype(bool).sum()),
            "clean_v2_removed_chars_total": int(pd.Series(out["clean_v2_removed_chars"]).fillna(0).astype(int).sum()),
        },
        "policy": {
            "data_version": "clean_v2",
            "locked_test_excluded": True,
            "al_rows_excluded": True,
            "branch": "gold300_only_relevance_readjudicated_v1",
        },
        "sha256": {
            "gold300_input": sha256_file(args.gold300),
            "clean_v2_input": sha256_file(args.clean_v2),
            "train_package": sha256_file(output_csv),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
