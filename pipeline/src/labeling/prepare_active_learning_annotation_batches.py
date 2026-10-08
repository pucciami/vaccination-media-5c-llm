from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

DEFAULT_STAGE1_CACHE = (
    CACHE_DIR
    / "datasets--Pucciami--qwen3-32b-relevance-tranche-output-v3"
    / "snapshots"
    / "2fa66635ae100be64b93a284fe8f218adac4b353"
)
DEFAULT_STAGE2_CACHE = (
    CACHE_DIR
    / "datasets--Pucciami--qwen3-32b-stage2-exploratory-output-v1"
    / "snapshots"
    / "823aad503ee1ac40451db06a0f5fcd5bd32c6544"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ranked active-learning candidate tables and Label Studio import batches from cached stage-1/stage-2 outputs."
    )
    parser.add_argument(
        "--stage1-tranche-csv",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "active_learning" / "media_active_learning_stage1_tranche_v1.csv",
    )
    parser.add_argument(
        "--stage2-subset-csv",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "active_learning" / "media_active_learning_stage2_exploratory_subset_v1.csv",
    )
    parser.add_argument("--stage1-cache-dir", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage2-cache-dir", type=Path, default=DEFAULT_STAGE2_CACHE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "active_learning",
    )
    parser.add_argument("--calc-batch-size", type=int, default=70)
    parser.add_argument("--broad-batch-size", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_stage1_outputs(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict[str, Any]] = []
    err_rows: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*predictions_chunk_*.jsonl")):
        pred_rows.extend(load_jsonl_rows(path))
    for path in sorted(cache_dir.glob("*errors_chunk_*.jsonl")):
        err_rows.extend(load_jsonl_rows(path))
    preds = pd.DataFrame(pred_rows)
    errs = pd.DataFrame(err_rows)
    if not preds.empty:
        preds["doc_id"] = preds["doc_id"].astype(str)
    if not errs.empty:
        errs["doc_id"] = errs["doc_id"].astype(str)
    return preds, errs


def load_stage2_outputs(cache_dir: Path) -> pd.DataFrame:
    pred_rows: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*predictions_chunk_*.jsonl")):
        pred_rows.extend(load_jsonl_rows(path))
    preds = pd.DataFrame(pred_rows)
    if not preds.empty:
        preds["doc_id"] = preds["doc_id"].astype(str)
    return preds


def normalize_presence(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v)]
        except json.JSONDecodeError:
            pass
    return [text]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_yes_no(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return "yes"
    if text in {"no", "false", "0"}:
        return "no"
    return ""


def valence_to_ls_choice(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    mapping = {
        "1": "+1",
        "1.0": "+1",
        "+1": "+1",
        "+1.0": "+1",
        "0": "0",
        "0.0": "0",
        "-1": "-1",
        "-1.0": "-1",
    }
    return mapping.get(text)


def add_reason(existing: str, reason: str) -> str:
    parts = [p for p in str(existing).split(";") if p]
    if reason not in parts:
        parts.append(reason)
    return ";".join(parts)


def merge_stage1_tranche(
    tranche_csv: Path, stage1_preds: pd.DataFrame, stage1_errs: pd.DataFrame
) -> pd.DataFrame:
    tranche = pd.read_csv(tranche_csv, dtype=str, keep_default_na=False)
    tranche["doc_id"] = tranche["doc_id"].astype(str)

    if not stage1_preds.empty:
        stage1_preds = stage1_preds.rename(columns={c: f"stage1_{c}" for c in stage1_preds.columns if c != "doc_id"})
        tranche = tranche.merge(stage1_preds, on="doc_id", how="left")
    if not stage1_errs.empty:
        stage1_errs = stage1_errs.rename(columns={c: f"stage1_err_{c}" for c in stage1_errs.columns if c != "doc_id"})
        tranche = tranche.merge(stage1_errs, on="doc_id", how="left")
    return tranche


def merge_stage2_subset(subset_csv: Path, stage2_preds: pd.DataFrame) -> pd.DataFrame:
    subset = pd.read_csv(subset_csv, dtype=str, keep_default_na=False)
    subset["doc_id"] = subset["doc_id"].astype(str)
    stage2_preds = stage2_preds.rename(columns={c: f"stage2_{c}" for c in stage2_preds.columns if c != "doc_id"})
    merged = subset.merge(stage2_preds, on="doc_id", how="left", validate="one_to_one")

    merged["stage2_c5_presence_norm"] = merged["stage2_c5_presence_d1"].apply(normalize_presence)
    merged["stage2_calc_present"] = merged["stage2_c5_presence_norm"].apply(lambda vals: "calculation" in vals)
    merged["stage2_calc_valence_norm"] = merged["stage2_valence_calculation_d1"].apply(valence_to_ls_choice)
    merged["stage2_relevance_norm"] = merged["stage2_relevance"].astype(str).str.strip().str.lower()
    merged["stage2_has_5c_norm"] = merged["stage2_has_relevant_5c"].apply(normalize_yes_no)
    merged["stage1_prompt_tokens_int"] = merged["stage1_usage_prompt_tokens"].apply(safe_int)
    merged["stage1_latency_float"] = merged["stage1_row_latency_seconds"].apply(safe_float)
    merged["stage2_latency_float"] = merged["stage2_row_latency_seconds"].apply(safe_float)
    merged["construct_count"] = merged["stage2_c5_presence_norm"].apply(len)
    return merged


def choose_calc_batch(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pool = df.copy()
    pool["calc_role"] = ""
    pool["batch_priority"] = ""

    selected_ids: set[str] = set()
    pieces: list[pd.DataFrame] = []

    def pick(frame: pd.DataFrame, n: int, role: str, sort_cols: list[str]) -> None:
        nonlocal pieces
        if n <= 0:
            return
        frame = frame[~frame["doc_id"].isin(selected_ids)].copy()
        if frame.empty:
            return
        frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(n).copy()
        frame["calc_role"] = role
        frame["batch_priority"] = role
        selected_ids.update(frame["doc_id"].tolist())
        pieces.append(frame)

    calc_cue = pool["calculation_cue"].astype(str).str.lower() == "true"
    calc_present = pool["stage2_calc_present"] == True

    pos = pool[calc_cue & calc_present & (pool["stage2_calc_valence_norm"] == "+1")]
    neu = pool[calc_cue & calc_present & (pool["stage2_calc_valence_norm"] == "0")]
    neg = pool[calc_cue & calc_present & (pool["stage2_calc_valence_norm"] == "-1")]
    hard_neg = pool[calc_cue & ~calc_present]

    pick(pos, 20, "calc_positive", ["stage2_latency_float", "stage1_prompt_tokens_int"])
    pick(neu, 20, "calc_neutral", ["stage2_latency_float", "stage1_prompt_tokens_int"])
    pick(neg, 10, "calc_negative", ["stage2_latency_float", "stage1_prompt_tokens_int"])
    pick(hard_neg, 20, "calc_hard_negative", ["stage2_latency_float", "stage1_prompt_tokens_int"])

    chosen = pd.concat(pieces, ignore_index=True) if pieces else pool.head(0).copy()

    if len(chosen) < target_size:
        remaining = pool[~pool["doc_id"].isin(selected_ids)].copy()
        if not remaining.empty:
            remaining = remaining.sample(frac=1, random_state=seed).copy()
            remaining["calc_role"] = "calc_fill"
            remaining["batch_priority"] = "calc_fill"
            chosen = pd.concat([chosen, remaining.head(target_size - len(chosen))], ignore_index=True)

    chosen = chosen.head(target_size).copy()
    chosen["al_batch"] = "calculation_first"
    chosen["al_batch_order"] = range(1, len(chosen) + 1)
    chosen["selection_summary"] = chosen.apply(
        lambda r: ";".join(
            [
                "calc_cue" if str(r.get("calculation_cue", "")).lower() == "true" else "",
                str(r.get("selection_flags", "")),
                str(r.get("calc_role", "")),
            ]
        ).strip(";"),
        axis=1,
    )
    return chosen


def choose_broad_batch(df: pd.DataFrame, excluded_ids: set[str], target_size: int, seed: int) -> pd.DataFrame:
    pool = df[~df["doc_id"].isin(excluded_ids)].copy()
    pool["broad_role"] = ""
    pool["batch_priority"] = ""
    selected_ids: set[str] = set()
    pieces: list[pd.DataFrame] = []

    def pick(frame: pd.DataFrame, n: int, role: str, sort_cols: list[str]) -> None:
        nonlocal pieces
        if n <= 0:
            return
        frame = frame[~frame["doc_id"].isin(selected_ids)].copy()
        if frame.empty:
            return
        frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(n).copy()
        frame["broad_role"] = role
        frame["batch_priority"] = role
        selected_ids.update(frame["doc_id"].tolist())
        pieces.append(frame)

    parser_edge = pool["subset_role"].astype(str) == "parser_edge_error"
    lead_false = pool["selection_flags"].astype(str).str.contains("lead_vax_anchor_false", regex=False)
    high_latency = pool["selection_flags"].astype(str).str.contains("high_latency", regex=False)
    high_prompt = pool["selection_flags"].astype(str).str.contains("high_prompt_tokens", regex=False)
    has_5c_no = pool["stage2_has_5c_norm"] == "no"
    multi_construct = pool["construct_count"] >= 2

    pick(pool[parser_edge], 10, "parser_edge", ["stage2_latency_float", "stage1_prompt_tokens_int"])
    pick(pool[lead_false & ~parser_edge], 25, "lead_false", ["stage2_latency_float", "stage1_prompt_tokens_int"])
    pick(pool[has_5c_no & ~lead_false & ~parser_edge], 20, "has_5c_no", ["stage2_latency_float", "stage1_prompt_tokens_int"])
    pick(pool[multi_construct & ~has_5c_no], 20, "multi_construct", ["construct_count", "stage2_latency_float"])
    pick(pool[high_latency & ~lead_false], 10, "high_latency", ["stage2_latency_float", "stage1_prompt_tokens_int"])
    pick(pool[high_prompt & ~lead_false], 10, "high_prompt", ["stage1_prompt_tokens_int", "stage2_latency_float"])

    chosen = pd.concat(pieces, ignore_index=True) if pieces else pool.head(0).copy()

    if len(chosen) < target_size:
        remaining = pool[~pool["doc_id"].isin(selected_ids)].copy()
        if not remaining.empty:
            # balance by year
            balanced: list[pd.DataFrame] = []
            year_groups = defaultdict(list)
            for _, row in remaining.iterrows():
                year_groups[str(row.get("year", ""))].append(row)
            rng = random.Random(seed)
            for rows in year_groups.values():
                rng.shuffle(rows)
            while sum(len(x) for x in balanced) < (target_size - len(chosen)) and any(year_groups.values()):
                for year in sorted(year_groups):
                    rows = year_groups[year]
                    if not rows:
                        continue
                    balanced.append(pd.DataFrame([rows.pop(0)]))
                    if sum(len(x) for x in balanced) >= (target_size - len(chosen)):
                        break
            if balanced:
                fill = pd.concat(balanced, ignore_index=True)
                fill["broad_role"] = "year_balanced_fill"
                fill["batch_priority"] = "year_balanced_fill"
                chosen = pd.concat([chosen, fill], ignore_index=True)

    chosen = chosen.head(target_size).copy()
    chosen["al_batch"] = "broader_hierarchy"
    chosen["al_batch_order"] = range(1, len(chosen) + 1)
    chosen["selection_summary"] = chosen.apply(
        lambda r: ";".join(
            [str(r.get("selection_flags", "")), str(r.get("broad_role", ""))]
        ).strip(";"),
        axis=1,
    )
    return chosen


def build_prediction_result(row: pd.Series) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
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

    relevance = str(row.get("stage2_relevance", "")).strip()
    if relevance:
        add_choices("relevance", [relevance])

    disease_primary = str(row.get("stage2_disease_1", "")).strip()
    if disease_primary and disease_primary.lower() not in {"none", "nan", "null"}:
        add_choices("disease_1", [disease_primary])

    has_relevant_5c = str(row.get("stage2_has_relevant_5c", "")).strip()
    if has_relevant_5c:
        add_choices("has_relevant_5c", [has_relevant_5c])

    constructs = normalize_presence(row.get("stage2_c5_presence_d1"))
    add_choices("c5_presence_d1", constructs)

    for field in [
        "stage2_valence_confidence_d1",
        "stage2_valence_complacency_d1",
        "stage2_valence_constraints_d1",
        "stage2_valence_calculation_d1",
        "stage2_valence_collective_d1",
    ]:
        ls_name = field.replace("stage2_", "")
        mapped = valence_to_ls_choice(row.get(field))
        if mapped is not None:
            add_choices(ls_name, [mapped])

    return results


def build_ls_tasks(df: pd.DataFrame, batch_name: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        task = {
            "data": {
                "doc_id": str(row["doc_id"]),
                "source_doc_id": str(row["source_doc_id"]),
                "title": row["title"],
                "text": row["body_text"] or row["text"],
                "selection_summary": row.get("selection_summary", ""),
                "selection_flags": row.get("selection_flags", ""),
                "subset_role": row.get("subset_role", ""),
                "al_batch": row.get("al_batch", batch_name),
                "calculation_cue": str(row.get("calculation_cue", "")),
                "stage1_relevance": row.get("stage1_relevance", ""),
            }
        }
        prediction_result = build_prediction_result(row)
        if prediction_result:
            task["predictions"] = [
                {
                    "model_version": "qwen_al_stage2_exploratory_v1",
                    "score": 0.0,
                    "result": prediction_result,
                }
            ]
        tasks.append(task)
    return tasks


def write_batch(df: pd.DataFrame, out_dir: Path, batch_name: str, manifest_extra: dict[str, Any]) -> None:
    json_path = out_dir / f"{batch_name}.json"
    csv_path = out_dir / f"{batch_name}.csv"
    manifest_path = out_dir / f"{batch_name}_manifest.json"

    tasks = build_ls_tasks(df, batch_name)
    json_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest = {
        "batch_name": batch_name,
        "n_rows": int(len(df)),
        "doc_ids": df["doc_id"].astype(str).tolist(),
        "selection_flag_counts": {
            k: int(v)
            for k, v in Counter(
                flag
                for value in df["selection_flags"].astype(str)
                for flag in value.split(";")
                if flag
            ).items()
        },
        "preannotations_included": True,
        "clean_input_rule": "title + main text only; no TEME",
        **manifest_extra,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stage1_preds, stage1_errs = load_stage1_outputs(args.stage1_cache_dir)
    stage2_preds = load_stage2_outputs(args.stage2_cache_dir)
    if stage2_preds.empty:
        raise RuntimeError("No cached stage-2 prediction chunks found.")

    _ = merge_stage1_tranche(args.stage1_tranche_csv, stage1_preds, stage1_errs)
    stage2_df = merge_stage2_subset(args.stage2_subset_csv, stage2_preds)

    candidate_table = stage2_df.copy()
    candidate_table["candidate_value_note"] = candidate_table.apply(
        lambda r: ";".join(
            p
            for p in [
                "calc_cue" if str(r.get("calculation_cue", "")).lower() == "true" else "",
                "parser_edge" if str(r.get("subset_role", "")) == "parser_edge_error" else "",
                "lead_false" if "lead_vax_anchor_false" in str(r.get("selection_flags", "")) else "",
                "high_prompt" if "high_prompt_tokens" in str(r.get("selection_flags", "")) else "",
                "high_latency" if "high_latency" in str(r.get("selection_flags", "")) else "",
                "year_control" if "year_balance_fill" in str(r.get("selection_flags", "")) else "",
                "stage2_calc_present" if bool(r.get("stage2_calc_present")) else "",
                f"stage2_has_5c_{r.get('stage2_has_5c_norm','')}" if str(r.get("stage2_has_5c_norm", "")) else "",
            ]
            if p
        ),
        axis=1,
    )
    candidate_table_path = args.out_dir / "media_active_learning_candidates_ranked_v1.csv"
    candidate_table.to_csv(candidate_table_path, index=False, encoding="utf-8-sig")

    calc_batch = choose_calc_batch(stage2_df, args.calc_batch_size, args.seed)
    broad_batch = choose_broad_batch(stage2_df, set(calc_batch["doc_id"].tolist()), args.broad_batch_size, args.seed + 1)

    write_batch(
        calc_batch,
        args.out_dir,
        "media_active_learning_calculation_batch_v1",
        {
            "batch_role": "calculation_first",
            "strict_calculation_boundary": True,
            "target_annotation_use": "dedicated Calculation enrichment wave",
        },
    )
    write_batch(
        broad_batch,
        args.out_dir,
        "media_active_learning_broad_batch_v1",
        {
            "batch_role": "broader_hierarchy",
            "target_annotation_use": "broader stage-2 and hierarchy active-learning wave",
        },
    )

    summary = {
        "candidate_table": str(candidate_table_path),
        "calc_batch_size": int(len(calc_batch)),
        "broad_batch_size": int(len(broad_batch)),
        "calc_role_counts": calc_batch["calc_role"].value_counts().to_dict(),
        "broad_role_counts": broad_batch["broad_role"].value_counts().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
