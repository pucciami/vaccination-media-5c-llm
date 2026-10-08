from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


CALCULATION_CUES = [
    "decide whether to vaccinate",
    "whether to vaccinate",
    "should i vaccinate",
    "should you vaccinate",
    "make the best decision",
    "decide for yourself",
    "weigh benefits and risks",
    "weigh risks and benefits",
    "risk-benefit",
    "pros and cons",
    "compare vaccines",
    "which vaccine",
    "ali se cepiti",
    "ali naj se cepi",
    "odlociti se za cepljenje",
    "odlocitev za cepljenje",
    "presoditi sami",
    "pretehtati koristi in tveganja",
    "pretehtati prednosti in tveganja",
    "koristi in tveganja",
    "prednosti in slabosti",
    "s katerim cepivom",
    "katero cepivo izbrati",
    "dilema s katerim se cepiti",
    "smiselno cepiti",
    "primerjava cepiv",
    "razlika med cepivi",
    "odmerek",
    "pozivitveni odmerek",
    "komu priporocajo",
    "za koga je primerno",
    "compare vaccine options",
    "morate presoditi sami",
    "naj vsak presodi",
    "odlocitev je na posamezniku",
    "patients should discuss with their doctor",
    "readers should consider",
    "balanced review",
    "risks and benefits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an exploratory targeted stage-2 active-learning subset from completed stage-1 tranche outputs."
    )
    parser.add_argument(
        "--tranche-csv",
        type=Path,
        default=Path("data/processed/active_learning/media_active_learning_stage1_tranche_v1.csv"),
    )
    parser.add_argument(
        "--stage1-output-repo",
        default="Pucciami/qwen3-32b-relevance-tranche-output-v3",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/active_learning/media_active_learning_stage2_exploratory_subset_v1.csv"),
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("data/processed/active_learning/media_active_learning_stage2_exploratory_subset_v1_manifest.json"),
    )
    parser.add_argument("--target-size", type=int, default=600)
    parser.add_argument("--max-parser-edge", type=int, default=20)
    parser.add_argument("--max-lead-false", type=int, default=150)
    parser.add_argument("--max-calculation", type=int, default=180)
    parser.add_argument("--max-high-prompt", type=int, default=120)
    parser.add_argument("--max-high-latency", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_stage1_repo_rows(repo_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    pred_files = sorted([f for f in files if "_predictions_chunk_" in f])
    err_files = sorted([f for f in files if "_errors_chunk_" in f])
    preds: list[dict[str, Any]] = []
    errs: list[dict[str, Any]] = []
    for name in pred_files:
        path = Path(api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name))
        preds.extend(read_jsonl(path))
    for name in err_files:
        path = Path(api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name))
        errs.extend(read_jsonl(path))
    return preds, errs


def load_tranche_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["doc_id"] = str(row["doc_id"])
            rows[row["doc_id"]] = row
    return rows


def text_has_calculation_cue(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(row.get(key, "")).lower()
        for key in ("title", "lead_text", "text", "model_text", "body_text")
    )
    return any(cue in haystack for cue in CALCULATION_CUES)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def main() -> None:
    args = parse_args()
    tranche_by_doc = load_tranche_rows(args.tranche_csv)
    pred_rows, err_rows = load_stage1_repo_rows(args.stage1_output_repo)

    successful: list[dict[str, Any]] = []
    parser_edge: list[dict[str, Any]] = []

    for pred in pred_rows:
        doc_id = str(pred["doc_id"])
        base = dict(tranche_by_doc[doc_id])
        merged = dict(base)
        merged.update({f"stage1_{k}": v for k, v in pred.items()})
        merged["subset_role"] = "stage2_candidate_success"
        merged["calculation_cue"] = str(text_has_calculation_cue(base))
        successful.append(merged)

    for err in err_rows:
        doc_id = str(err["doc_id"])
        if doc_id not in tranche_by_doc:
            continue
        raw_excerpt = str(err.get("raw_text_excerpt", "")).lower()
        if "not relevant" not in raw_excerpt:
            continue
        base = dict(tranche_by_doc[doc_id])
        merged = dict(base)
        merged.update({f"stage1_{k}": v for k, v in err.items()})
        merged["stage1_relevance"] = "not_relevant_variant"
        merged["subset_role"] = "parser_edge_error"
        merged["calculation_cue"] = str(text_has_calculation_cue(base))
        parser_edge.append(merged)

    selected: dict[str, dict[str, Any]] = {}

    def add_rows(rows: list[dict[str, Any]], reason: str, limit: int | None = None) -> None:
        added = 0
        for row in rows:
            doc_id = row["doc_id"]
            if doc_id in selected:
                current = selected[doc_id].get("selection_flags", "")
                flags = [flag for flag in current.split(";") if flag]
                if reason not in flags:
                    flags.append(reason)
                    selected[doc_id]["selection_flags"] = ";".join(flags)
                continue
            row = dict(row)
            row["selection_flags"] = reason
            selected[doc_id] = row
            added += 1
            if limit is not None and added >= limit:
                break

    parser_edge_sorted = sorted(parser_edge, key=lambda r: safe_float(r.get("stage1_row_latency_seconds")), reverse=True)
    add_rows(parser_edge_sorted, "parser_edge_error", args.max_parser_edge)

    lead_false_rows = [
        row for row in successful if str(row.get("lead_vax_anchor", "")).strip().lower() == "false"
    ]
    lead_false_rows.sort(key=lambda r: safe_float(r.get("stage1_usage_prompt_tokens")), reverse=True)
    add_rows(lead_false_rows, "lead_vax_anchor_false", args.max_lead_false)

    calc_rows = [row for row in successful if str(row.get("calculation_cue", "")).lower() == "true"]
    calc_rows.sort(
        key=lambda r: (
            safe_float(r.get("stage1_usage_prompt_tokens")),
            safe_float(r.get("stage1_row_latency_seconds")),
        ),
        reverse=True,
    )
    add_rows(calc_rows, "calculation_cue", args.max_calculation)

    high_prompt_rows = sorted(
        successful,
        key=lambda r: safe_int(r.get("stage1_usage_prompt_tokens")),
        reverse=True,
    )
    add_rows(high_prompt_rows, "high_prompt_tokens", args.max_high_prompt)

    high_latency_rows = sorted(
        successful,
        key=lambda r: safe_float(r.get("stage1_row_latency_seconds")),
        reverse=True,
    )
    add_rows(high_latency_rows, "high_latency", args.max_high_latency)

    remaining = [row for row in successful if row["doc_id"] not in selected]
    year_groups: dict[str, list[dict[str, Any]]] = {}
    for row in remaining:
        year_groups.setdefault(str(row.get("year", "")), []).append(row)
    rng = random.Random(args.seed)
    for rows in year_groups.values():
        rng.shuffle(rows)

    while len(selected) < args.target_size and any(year_groups.values()):
        for year in sorted(year_groups):
            rows = year_groups[year]
            if not rows:
                continue
            row = rows.pop(0)
            add_rows([row], "year_balance_fill", 1)
            if len(selected) >= args.target_size:
                break

    final_rows = list(selected.values())
    final_rows.sort(key=lambda r: (r.get("subset_role", ""), r.get("year", ""), r.get("doc_id", "")))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    if final_rows:
        seen: set[str] = set()
        for row in final_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    selection_counter = Counter()
    role_counter = Counter()
    year_counter = Counter()
    for row in final_rows:
        role_counter[row["subset_role"]] += 1
        year_counter[str(row.get("year", ""))] += 1
        for flag in str(row.get("selection_flags", "")).split(";"):
            if flag:
                selection_counter[flag] += 1

    manifest = {
        "stage1_output_repo": args.stage1_output_repo,
        "tranche_csv": str(args.tranche_csv),
        "out_csv": str(args.out_csv),
        "target_size": int(args.target_size),
        "n_stage1_success_rows_available": int(len(successful)),
        "n_stage1_parser_edge_rows_available": int(len(parser_edge)),
        "n_final_rows": int(len(final_rows)),
        "subset_role": "exploratory_active_learning_stage2_subset",
        "production_inference": False,
        "full_corpus_prevalence_estimate_allowed": False,
        "selection_priority": [
            "parser_edge_error",
            "lead_vax_anchor_false",
            "calculation_cue",
            "high_prompt_tokens",
            "high_latency",
            "year_balance_fill",
        ],
        "selection_counts": dict(selection_counter),
        "role_counts": dict(role_counter),
        "year_counts_top": dict(year_counter.most_common(12)),
        "note": (
            "Exploratory active-learning subset only. Do not use downstream outputs from this subset to estimate "
            "full-corpus prevalence."
        ),
    }
    args.out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved exploratory stage-2 subset CSV: {args.out_csv}")
    print(f"Saved manifest: {args.out_manifest}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
