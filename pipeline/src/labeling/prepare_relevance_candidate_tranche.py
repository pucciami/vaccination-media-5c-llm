from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_POOL = BASE_DIR / "data" / "processed" / "active_learning" / "media_active_learning_pool_v1.csv"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "processed" / "active_learning"

VAX_PATTERN = re.compile(
    r"\b(cepiv|cepivo|cepljenj|cepljenje|cepljenju|cepljeni|cepite|imunizacij|vaccin|vaccine|immuniz)\w*\b",
    re.IGNORECASE,
)
OTHER_OUTLET_LABEL = "__other_outlets__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a smaller active-learning stage-1 screening tranche from the post-gold media pool."
    )
    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tranche-name", default="media_active_learning_stage1_tranche_v1")
    parser.add_argument("--title-anchor-max", type=int, default=3500)
    parser.add_argument("--lead-anchor-only", type=int, default=700)
    parser.add_argument("--hard-negative-controls", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_sample(df: pd.DataFrame, n: int, seed: int, group_cols: list[str]) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0).copy()
    if len(df) <= n:
        return df.copy()

    work = df.copy()
    work["_group"] = work[group_cols].astype(str).agg(" | ".join, axis=1)
    group_sizes = work["_group"].value_counts().to_dict()
    total = len(work)

    picks: list[pd.DataFrame] = []
    allocated = 0
    for i, (group_name, group_df) in enumerate(work.groupby("_group", sort=False), start=1):
        remaining_groups = len(group_sizes) - i + 1
        target = max(1, round(n * len(group_df) / total))
        max_allowed = n - allocated - max(remaining_groups - 1, 0)
        take = min(len(group_df), max(1, max_allowed), target)
        picks.append(group_df.sample(n=take, random_state=seed + i))
        allocated += take

    sampled = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["doc_id"])
    if len(sampled) < n:
        extra = work[~work["doc_id"].isin(sampled["doc_id"])].sample(
            n=min(n - len(sampled), len(work) - len(sampled)),
            random_state=seed + 999,
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.head(n).drop(columns="_group", errors="ignore")
    return sampled


def add_sampling_balance_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["outlet"] = out["outlet"].fillna("").astype(str).str.strip()
    top_outlets = out["outlet"].replace("", OTHER_OUTLET_LABEL).value_counts().head(12).index.tolist()
    out["outlet_balance"] = out["outlet"].where(out["outlet"].isin(top_outlets), OTHER_OUTLET_LABEL)
    out["year_balance"] = out["year"].astype(str)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(args.pool_path, dtype=str, keep_default_na=False)
    pool["year"] = pool["year"].astype(str)
    pool["title"] = pool["title"].fillna("").astype(str)
    pool["body_text"] = pool["body_text"].fillna("").astype(str)
    pool["lead_text"] = pool["body_text"].str.slice(0, 600)

    pool["title_vax_anchor"] = pool["title"].str.contains(VAX_PATTERN, na=False)
    pool["lead_vax_anchor"] = pool["lead_text"].str.contains(VAX_PATTERN, na=False)
    pool["body_vax_anchor"] = pool["body_text"].str.contains(VAX_PATTERN, na=False)
    pool = add_sampling_balance_columns(pool)

    title_anchor = pool[pool["title_vax_anchor"]].copy()
    title_anchor = stratified_sample(title_anchor, args.title_anchor_max, args.seed + 500, ["year_balance", "outlet_balance"])
    title_anchor["selection_reason"] = "title_vax_anchor_sample"

    lead_only = pool[(~pool["title_vax_anchor"]) & (pool["lead_vax_anchor"])].copy()
    lead_only = stratified_sample(lead_only, args.lead_anchor_only, args.seed, ["year_balance", "outlet_balance"])
    lead_only["selection_reason"] = "lead_vax_anchor_sample"

    hard_negative = pool[(~pool["title_vax_anchor"]) & (~pool["lead_vax_anchor"])].copy()
    hard_negative = stratified_sample(hard_negative, args.hard_negative_controls, args.seed + 1000, ["year_balance", "outlet_balance"])
    hard_negative["selection_reason"] = "no_anchor_control_sample"

    tranche = pd.concat([title_anchor, lead_only, hard_negative], ignore_index=True)
    tranche = tranche.drop_duplicates(subset=["doc_id"]).copy()
    tranche["tranche_name"] = args.tranche_name

    csv_path = args.out_dir / f"{args.tranche_name}.csv"
    parquet_path = args.out_dir / f"{args.tranche_name}.parquet"
    manifest_path = args.out_dir / f"{args.tranche_name}_manifest.json"

    tranche.to_csv(csv_path, index=False, encoding="utf-8")
    tranche.to_parquet(parquet_path, index=False)

    manifest = {
        "tranche_name": args.tranche_name,
        "source_pool_path": str(args.pool_path.relative_to(BASE_DIR)),
        "n_total": int(len(tranche)),
        "selection_counts": {k: int(v) for k, v in tranche["selection_reason"].value_counts().to_dict().items()},
        "selection_logic": {
            "primary": "title and lead lexical vaccination anchors",
            "secondary_balance": ["year", "outlet"],
            "bucket_hint_role": "descriptive only; not used as a primary sampling signal",
        },
        "bucket_counts": {k: int(v) for k, v in tranche["bucket_hint"].value_counts().to_dict().items()},
        "year_counts": {str(k): int(v) for k, v in tranche["year"].value_counts().sort_index().to_dict().items()},
        "outlet_counts_head": {str(k): int(v) for k, v in tranche["outlet"].replace("", OTHER_OUTLET_LABEL).value_counts().head(20).to_dict().items()},
        "title_anchor_sample_count": int(len(title_anchor)),
        "lead_anchor_sample_count": int(len(lead_only)),
        "hard_negative_control_count": int(len(hard_negative)),
        "seed": int(args.seed),
        "clean_input_rule": "title + main text only; excludes locked 400 and does not expose TEME to model input",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote tranche csv: {csv_path}")
    print(f"Wrote tranche parquet: {parquet_path}")
    print(f"Wrote tranche manifest: {manifest_path}")
    print(f"Total tranche rows: {len(tranche)}")


if __name__ == "__main__":
    main()
