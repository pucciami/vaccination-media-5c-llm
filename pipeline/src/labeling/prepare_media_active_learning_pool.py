from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.common.media_text import build_model_text

DOCS_PATH = BASE_DIR / "data" / "interim" / "documents_media.parquet"
LOCKED_TRAIN_PATH = BASE_DIR / "data" / "locked" / "gold300_train" / "gold300_train_adjudicated.csv"
LOCKED_TEST_PATH = BASE_DIR / "data" / "locked" / "gold100" / "gold100_test_gold_labels.csv"

OUT_DIR = BASE_DIR / "data" / "processed" / "active_learning"
OUT_PARQUET = OUT_DIR / "media_active_learning_pool_v1.parquet"
OUT_CSV = OUT_DIR / "media_active_learning_pool_v1.csv"
OUT_MANIFEST = OUT_DIR / "media_active_learning_pool_v1_manifest.json"

PATTERNS = {
    "measles": re.compile(r"\b(ošpice|ospice|measles)\b", re.IGNORECASE),
    "covid19": re.compile(r"\b(covid|koronavirus|sars\-?cov\-?2)\b", re.IGNORECASE),
    "influenza": re.compile(r"\b(gripa|influenza)\b", re.IGNORECASE),
    "hpv": re.compile(r"\b(hpv|papilom)\b", re.IGNORECASE),
}


def bucket_row(topics: str, title: str, text: str) -> str:
    snippet = f"{topics}\n{title}\n{text[:800]}"
    for bucket, pattern in PATTERNS.items():
        if pattern.search(snippet):
            return bucket
    if re.search(r"\b(cepljen|cepljenj|cepiv|cepljenju|cepljenje)\b", snippet, re.IGNORECASE):
        return "vax_unspecified"
    return "other"


def extract_clean_body_text(title: str, combined_text: str) -> str:
    model_text = build_model_text(title, combined_text)
    if "\n\n" in model_text:
        body_text = model_text.split("\n\n", 1)[1].strip()
    else:
        body_text = ""

    title_text = str(title or "").strip()
    while title_text and body_text.startswith(title_text):
        body_text = body_text[len(title_text):].lstrip("\n").strip()

    return body_text


def load_excluded_source_doc_ids() -> set[str]:
    train_df = pd.read_csv(LOCKED_TRAIN_PATH, usecols=["source_doc_id"])
    test_df = pd.read_csv(LOCKED_TEST_PATH, usecols=["source_doc_id"])
    train_ids = set(train_df["source_doc_id"].dropna().astype(str))
    test_ids = set(test_df["source_doc_id"].dropna().astype(str))
    return train_ids | test_ids


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DOCS_PATH).copy()
    df["doc_id"] = df["doc_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["topics"] = df["topics"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["outlet"] = df["outlet"].fillna("").astype(str)

    excluded_source_doc_ids = load_excluded_source_doc_ids()
    original_count = len(df)

    df = df[~df["doc_id"].isin(excluded_source_doc_ids)].copy()
    after_exclusion_count = len(df)

    published_dt = pd.to_datetime(df["published_at"], errors="coerce")
    df["year"] = published_dt.dt.year.fillna(0).astype(int)
    df["bucket_hint"] = df.apply(
        lambda row: bucket_row(row["topics"], row["title"], row["text"]),
        axis=1,
    )

    df["source_doc_id"] = df["doc_id"]
    df["selection_reason"] = "eligible_pool_post_gold_exclusion"
    df["pool_version"] = "media_active_learning_pool_v1"
    df["model_text"] = df.apply(lambda row: build_model_text(row["title"], row["text"]), axis=1)
    df["body_text"] = df.apply(lambda row: extract_clean_body_text(row["title"], row["text"]), axis=1)

    ordered_columns = [
        "doc_id",
        "source_doc_id",
        "published_at",
        "year",
        "outlet",
        "title",
        "topics",
        "bucket_hint",
        "selection_reason",
        "pool_version",
        "model_text",
        "body_text",
        "text",
    ]
    pool_df = df[ordered_columns].copy()

    pool_df.to_parquet(OUT_PARQUET, index=False)
    pool_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    manifest = {
        "pool_version": "media_active_learning_pool_v1",
        "source_documents_path": str(DOCS_PATH.relative_to(BASE_DIR)),
        "locked_train_path": str(LOCKED_TRAIN_PATH.relative_to(BASE_DIR)),
        "locked_test_path": str(LOCKED_TEST_PATH.relative_to(BASE_DIR)),
        "original_documents_count": int(original_count),
        "excluded_gold_source_doc_ids_count": int(len(excluded_source_doc_ids)),
        "eligible_documents_count": int(after_exclusion_count),
        "columns": ordered_columns,
        "bucket_counts": {k: int(v) for k, v in pool_df["bucket_hint"].value_counts().to_dict().items()},
        "year_counts_head": {str(k): int(v) for k, v in pool_df["year"].value_counts().sort_index().head(20).to_dict().items()},
        "year_counts_tail": {str(k): int(v) for k, v in pool_df["year"].value_counts().sort_index().tail(10).to_dict().items()},
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote pool parquet: {OUT_PARQUET}")
    print(f"Wrote pool csv: {OUT_CSV}")
    print(f"Wrote manifest: {OUT_MANIFEST}")
    print(f"Original documents: {original_count}")
    print(f"Eligible documents after gold exclusion: {after_exclusion_count}")


if __name__ == "__main__":
    main()
