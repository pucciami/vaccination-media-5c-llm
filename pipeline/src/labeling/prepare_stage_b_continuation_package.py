import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/processed/clean_v2_al_relevance_readjudicated_v1/media_train_enriched_clean_v2_al159.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/clean_v2_stage_b_al159_replay90_v1")
CONSTRUCTS = [
    "confidence",
    "complacency",
    "constraints",
    "calculation",
    "collective_responsibility",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Stage B continuation package: all AL159 rows plus a "
            "deterministic gold300 replay slice to reduce forgetting from the Stage A adapter."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--replay-size", type=int, default=90)
    parser.add_argument("--not-relevant-replay", type=int, default=30)
    parser.add_argument("--relevant-no5c-replay", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260417)
    return parser.parse_args()


def stable_key(seed: int, *parts: Any) -> str:
    text = ":".join([str(seed), *[str(part) for part in parts]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_presence(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text in {"[]", "null", "None"}:
        return []
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return [text]


def primary_construct(value: Any) -> str:
    presence = parse_presence(value)
    if not presence:
        return "none"
    for construct in CONSTRUCTS:
        if construct in presence:
            return construct
    return sorted(presence)[0]


def take_deterministic(df: pd.DataFrame, n: int, seed: int, salt: str) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0).copy()
    ordered = df.copy()
    ordered["_stage_b_key"] = ordered["doc_id"].astype(str).apply(lambda value: stable_key(seed, salt, value))
    return ordered.sort_values("_stage_b_key").head(min(n, len(ordered))).drop(columns=["_stage_b_key"])


def proportional_replay(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0).copy()
    work = df.copy()
    work["_replay_stratum"] = (
        work["ann1_disease_1"].fillna("").replace("", "null")
        + "|"
        + work["ann1_c5_presence_d1"].apply(primary_construct)
    )
    strata = []
    for stratum, group in work.groupby("_replay_stratum", sort=True):
        exact = n * len(group) / len(work)
        strata.append(
            {
                "stratum": stratum,
                "size": len(group),
                "base": int(exact),
                "remainder": exact - int(exact),
            }
        )
    remaining = n - sum(item["base"] for item in strata)
    for item in sorted(strata, key=lambda row: (-row["remainder"], row["stratum"]))[:remaining]:
        item["base"] += 1

    pieces = []
    for item in strata:
        group = work[work["_replay_stratum"] == item["stratum"]].copy()
        pieces.append(take_deterministic(group, item["base"], seed, f"replay:{item['stratum']}"))
    selected = pd.concat(pieces, ignore_index=True) if pieces else work.head(0).copy()

    if len(selected) < n:
        selected_ids = set(selected["doc_id"].astype(str))
        remainder = work[~work["doc_id"].astype(str).isin(selected_ids)].copy()
        selected = pd.concat(
            [selected, take_deterministic(remainder, n - len(selected), seed, "replay:topup")],
            ignore_index=True,
        )
    if len(selected) > n:
        selected = take_deterministic(selected, n, seed, "replay:trim")
    return selected.drop(columns=[col for col in selected.columns if col.startswith("_replay_")], errors="ignore")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if "package_origin" not in df.columns:
        raise ValueError("Input must include package_origin.")
    if df["doc_id"].astype(str).duplicated().any():
        raise ValueError("Input contains duplicate doc_id values.")

    al = df[df["package_origin"] == "active_learning_al159"].copy()
    gold = df[df["package_origin"] == "gold300_locked"].copy()
    if len(al) != 159:
        raise ValueError(f"Expected 159 AL rows, found {len(al)}.")
    if len(gold) != 300:
        raise ValueError(f"Expected 300 gold300 rows, found {len(gold)}.")

    not_relevant = gold[gold["ann1_relevance"] == "not_relevant"].copy()
    relevant_no5c = gold[
        (gold["ann1_relevance"] == "relevant") & (gold["ann1_has_relevant_5c"] == "no")
    ].copy()
    relevant_yes5c = gold[
        (gold["ann1_relevance"] == "relevant") & (gold["ann1_has_relevant_5c"] == "yes")
    ].copy()

    replay_not_relevant = take_deterministic(
        not_relevant,
        args.not_relevant_replay,
        args.seed,
        "not_relevant",
    )
    replay_no5c = take_deterministic(
        relevant_no5c,
        args.relevant_no5c_replay,
        args.seed,
        "relevant_no5c",
    )
    remaining_replay = args.replay_size - len(replay_not_relevant) - len(replay_no5c)
    replay_yes5c = proportional_replay(relevant_yes5c, remaining_replay, args.seed)

    replay = pd.concat([replay_not_relevant, replay_no5c, replay_yes5c], ignore_index=True)
    replay = replay.assign(stage_b_component="gold300_replay")
    al = al.assign(stage_b_component="active_learning_al159")
    stage_b = pd.concat([al, replay], ignore_index=True)
    stage_b["_order_key"] = stage_b.apply(
        lambda row: stable_key(args.seed, row["stage_b_component"], row["doc_id"]),
        axis=1,
    )
    stage_b = stage_b.sort_values("_order_key").drop(columns=["_order_key"]).reset_index(drop=True)

    package_path = output_dir / "media_train_stage_b_al159_replay90.csv"
    split_path = output_dir / "media_train_stage_b_al159_replay90_split.csv"
    manifest_path = output_dir / "media_train_stage_b_al159_replay90_manifest.json"
    stage_b.to_csv(package_path, index=False)
    stage_b[
        [
            "doc_id",
            "labelstudio_task_id",
            "source_doc_id",
            "year",
            "published_at",
            "outlet",
            "title",
            "provisional_disease",
            "disease_group",
            "year_bucket",
            "split",
            "stage_b_component",
        ]
    ].to_csv(split_path, index=False)

    def counts(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "n": int(len(frame)),
            "relevance": frame["ann1_relevance"].value_counts(dropna=False).to_dict(),
            "has_relevant_5c": frame["ann1_has_relevant_5c"].replace("", "null").value_counts(dropna=False).to_dict(),
            "construct_presence": {
                construct: int(frame["ann1_c5_presence_d1"].apply(lambda value, c=construct: c in parse_presence(value)).sum())
                for construct in CONSTRUCTS
            },
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "outputs": {
            "train_package": str(package_path),
            "split_assignment": str(split_path),
        },
        "seed": args.seed,
        "policy": {
            "stage": "B",
            "purpose": "continue Stage A adapter on AL159 with deterministic gold300 replay",
            "locked_test_excluded": True,
            "all_al_rows_included": True,
            "replay_size": args.replay_size,
            "not_relevant_replay_target": args.not_relevant_replay,
            "relevant_no5c_replay_target": args.relevant_no5c_replay,
        },
        "counts": {
            "active_learning_al159": counts(al),
            "gold300_replay": counts(replay),
            "stage_b_total": counts(stage_b),
        },
        "replay_doc_ids": sorted(replay["doc_id"].astype(str).tolist(), key=lambda value: int(value) if value.isdigit() else value),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved Stage B package: {package_path}")
    print(f"Saved Stage B split: {split_path}")
    print(f"Saved Stage B manifest: {manifest_path}")
    print(f"Rows: AL={len(al)}, replay={len(replay)}, total={len(stage_b)}")


if __name__ == "__main__":
    main()
