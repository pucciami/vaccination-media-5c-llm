from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.common.media_text import build_model_text
DEFAULT_POOL = BASE_DIR / "data" / "processed" / "active_learning" / "media_active_learning_pool_v1.parquet"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "processed" / "active_learning"

DEFAULT_QUOTA = {
    "measles": 13,
    "covid19": 23,
    "influenza": 20,
    "hpv": 13,
    "vax_unspecified": 20,
    "other": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh Label Studio import batch for the media active-learning workflow."
    )
    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-name", default="media_active_learning_batch_v1_100")
    parser.add_argument("--n-total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Optional normalized CSV or JSONL with model predictions keyed by doc_id/source_doc_id.",
    )
    return parser.parse_args()


def load_pool(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["doc_id"] = df["doc_id"].astype(str)
    df["source_doc_id"] = df["source_doc_id"].astype(str)
    if "body_text" not in df.columns:
        df["body_text"] = df.apply(
            lambda row: str(build_model_text(row.get("title", ""), row.get("text", ""))).split("\n\n", 1)[1]
            if "\n\n" in str(build_model_text(row.get("title", ""), row.get("text", "")))
            else "",
            axis=1,
        )
    return df


def load_predictions(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)

    if "doc_id" not in df.columns:
        raise ValueError("Predictions file must contain doc_id.")
    df["doc_id"] = df["doc_id"].astype(str)
    return df


def sample_bucket(sub: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    sub = sub.copy()
    sub = sub[sub["text"].astype(str).str.len() > 0]
    if len(sub) == 0:
        return sub

    years = [y for y in sorted(sub["year"].dropna().astype(int).unique().tolist()) if int(y) > 0]
    if not years:
        years = [0]

    per_year = max(1, n // max(1, len(years)))
    outlet_counts: Counter[str] = Counter()
    picked_idx: list[int] = []

    def pick_from_frame(frame: pd.DataFrame, k: int, outlet_cap: int) -> list[int]:
        if k <= 0 or len(frame) == 0:
            return []
        idxs = frame.index.tolist()
        rng.shuffle(idxs)
        out: list[int] = []
        for ix in idxs:
            outlet = str(frame.at[ix, "outlet"])
            if outlet_counts[outlet] < outlet_cap:
                out.append(ix)
                outlet_counts[outlet] += 1
                if len(out) >= k:
                    break
        return out

    outlet_cap = 15
    for year in years:
        frame = sub[sub["year"] == year] if year != 0 else sub
        picked_idx.extend(pick_from_frame(frame, per_year, outlet_cap))

    if len(picked_idx) < n:
        remaining = sub.drop(index=picked_idx, errors="ignore")
        picked_idx.extend(pick_from_frame(remaining, n - len(picked_idx), outlet_cap))

    while len(picked_idx) < n and outlet_cap < 1000:
        outlet_cap += 5
        remaining = sub.drop(index=picked_idx, errors="ignore")
        picked_idx.extend(pick_from_frame(remaining, n - len(picked_idx), outlet_cap))

    picked = sub.loc[picked_idx].copy()
    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed)
    return picked


def normalize_construct_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except json.JSONDecodeError:
            pass
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def normalize_textarea(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    return [text] if text else []


def first_non_empty(row: pd.Series, *keys: str):
    for key in keys:
        if key in row.index:
            value = row[key]
            if value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip():
                return value
    return None


def build_prediction_result(row: pd.Series) -> list[dict]:
    results: list[dict] = []
    to_name = "article_text"

    def add_choices(from_name: str, choices: list[str]) -> None:
        if not choices:
            return
        results.append(
            {
                "id": str(uuid.uuid4()),
                "from_name": from_name,
                "to_name": to_name,
                "type": "choices",
                "value": {"choices": choices},
            }
        )

    def add_text(from_name: str, texts: list[str]) -> None:
        if not texts:
            return
        results.append(
            {
                "id": str(uuid.uuid4()),
                "from_name": from_name,
                "to_name": to_name,
                "type": "textarea",
                "value": {"text": texts},
            }
        )

    relevance = first_non_empty(row, "pred_relevance", "relevance")
    if relevance:
        add_choices("relevance", [str(relevance)])

    disease_primary = first_non_empty(row, "pred_disease_1", "disease_1")
    has_relevant_5c = first_non_empty(row, "pred_has_relevant_5c", "has_relevant_5c")
    if has_relevant_5c:
        add_choices("has_relevant_5c", [str(has_relevant_5c)])

    if disease_primary:
        add_choices("disease_1", [str(disease_primary)])
    add_text(
        "disease_1_other_text",
        normalize_textarea(first_non_empty(row, "pred_disease_1_other_text", "disease_1_other_text")),
    )

    constructs = normalize_construct_list(first_non_empty(row, "pred_c5_presence_d1", "c5_presence_d1"))
    add_choices("c5_presence_d1", constructs)

    for field in [
        "valence_confidence_d1",
        "valence_complacency_d1",
        "valence_constraints_d1",
        "valence_calculation_d1",
        "valence_collective_d1",
    ]:
        value = first_non_empty(row, f"pred_{field}", field)
        if value is not None and str(value).strip():
            add_choices(field, [str(value)])

    has_disease_2 = first_non_empty(row, "pred_has_disease_2", "has_disease_2")
    has_disease_3 = first_non_empty(row, "pred_has_disease_3", "has_disease_3")
    annotator_uncertainty = first_non_empty(row, "pred_annotator_uncertainty", "annotator_uncertainty")
    if has_disease_2:
        add_choices("has_disease_2", [str(has_disease_2)])
    if has_disease_3:
        add_choices("has_disease_3", [str(has_disease_3)])
    if annotator_uncertainty:
        add_choices("annotator_uncertainty", [str(annotator_uncertainty)])

    return results


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(args.pool_path)
    if args.n_total != sum(DEFAULT_QUOTA.values()):
        raise ValueError("This starter batch is currently defined for 100 tasks with fixed quotas.")

    sampled_parts = []
    for bucket, quota in DEFAULT_QUOTA.items():
        subset = pool[pool["bucket_hint"] == bucket].copy()
        picked = sample_bucket(subset, quota, args.seed + sum(ord(ch) for ch in bucket))
        if len(picked) < quota:
            raise ValueError(f"Bucket {bucket} only produced {len(picked)} tasks, expected {quota}.")
        picked["selection_reason"] = "diversity_stratified_post_gold_exclusion"
        sampled_parts.append(picked)

    batch_df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if batch_df["doc_id"].nunique() != len(batch_df):
        raise ValueError("Duplicate doc_id values detected in the batch.")

    predictions_df = load_predictions(args.predictions_path)
    if predictions_df is not None:
        batch_df = batch_df.merge(predictions_df, on="doc_id", how="left", validate="one_to_one")

    tasks = []
    for _, row in batch_df.iterrows():
        task = {
            "data": {
                "doc_id": row["doc_id"],
                "source_doc_id": row["source_doc_id"],
                "title": row["title"],
                "text": row["body_text"],
            }
        }

        if predictions_df is not None:
            prediction_result = build_prediction_result(row)
            if prediction_result:
                task["predictions"] = [
                    {
                        "model_version": str(first_non_empty(row, "model_version", "pred_model_version") or "external_preannotation"),
                        "score": float(first_non_empty(row, "prediction_score", "pred_score") or 0.0),
                        "result": prediction_result,
                    }
                ]

        tasks.append(task)

    json_path = args.out_dir / f"{args.batch_name}.json"
    csv_path = args.out_dir / f"{args.batch_name}.csv"
    manifest_path = args.out_dir / f"{args.batch_name}_manifest.json"

    json_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    batch_df.to_csv(csv_path, index=False, encoding="utf-8")

    manifest = {
        "batch_name": args.batch_name,
        "pool_path": str(args.pool_path.relative_to(BASE_DIR)),
        "predictions_path": str(args.predictions_path.relative_to(BASE_DIR)) if args.predictions_path else None,
        "n_total": int(len(batch_df)),
        "seed": int(args.seed),
        "selection_strategy": "diversity_stratified_post_gold_exclusion",
        "preannotations_included": predictions_df is not None,
        "bucket_counts": {k: int(v) for k, v in batch_df["bucket_hint"].value_counts().to_dict().items()},
        "outlet_counts_head": {k: int(v) for k, v in batch_df["outlet"].value_counts().head(15).to_dict().items()},
        "year_counts": {str(k): int(v) for k, v in batch_df["year"].value_counts().sort_index().to_dict().items()},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote Label Studio batch json: {json_path}")
    print(f"Wrote batch audit csv: {csv_path}")
    print(f"Wrote batch manifest: {manifest_path}")
    print(f"Preannotations included: {predictions_df is not None}")


if __name__ == "__main__":
    main()
