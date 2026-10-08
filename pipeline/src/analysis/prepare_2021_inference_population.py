from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from longitudinal_2021_common import (
    COVID_REGEX,
    DEFAULT_LEXICAL_SEARCH_CHARS,
    JANSSEN_REGEX,
    VACCINATION_REGEX,
    add_time_fields,
    lexical_flags,
    sha256_file,
    summarize_counts,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "clean_v2" / "documents_media_clean_v2.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "longitudinal_2021"
DEFAULT_REPORT_DIR = ROOT / "reports" / "longitudinal_2021"


POPULATIONS = {
    "longitudinal_2021_full_source": "All clean-v2 source articles published in 2021.",
    "longitudinal_2021_covid_vaccination_candidate": "Full-year lexical COVID/coronavirus/SARS-CoV-2 AND vaccination candidate subset.",
    "longitudinal_2021_aug_dec_covid_vaccination_candidate": "Aug-Dec 2021 lexical COVID vaccination event-window candidate subset.",
    "longitudinal_2021_janssen_vaccination_context": "Full-year lexical Janssen/J&J AND vaccination context subset.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local-only 2021 clean-v2 inference populations for the Eurosurveillance longitudinal application."
    )
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument(
        "--lexical-search-chars",
        type=int,
        default=DEFAULT_LEXICAL_SEARCH_CHARS,
        help="Search title plus this many leading clean-v2 body characters for candidate prefilters; use 0 for full text.",
    )
    return parser.parse_args()


def population_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    covid_vax = df["covid_vaccination_lexical"]
    janssen_vax = df["janssen_vaccination_lexical"]
    published = pd.to_datetime(df["published_at"], errors="coerce")
    aug_dec = (published >= pd.Timestamp("2021-08-01")) & (published <= pd.Timestamp("2021-12-31 23:59:59"))
    return {
        "longitudinal_2021_full_source": df.copy(),
        "longitudinal_2021_covid_vaccination_candidate": df.loc[covid_vax].copy(),
        "longitudinal_2021_aug_dec_covid_vaccination_candidate": df.loc[covid_vax & aug_dec].copy(),
        "longitudinal_2021_janssen_vaccination_context": df.loc[janssen_vax].copy(),
    }


def write_population(name: str, df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"{name}_input.csv"
    metadata_path = output_dir / f"{name}_metadata.csv"

    input_cols = ["doc_id", "source_doc_id", "model_text"]
    metadata_cols = [
        "doc_id",
        "source_doc_id",
        "published_at",
        "date",
        "year",
        "month",
        "week_start",
        "iso_year",
        "iso_week",
        "outlet",
        "title",
        "covid_lexical",
        "vaccination_lexical",
        "janssen_lexical",
        "janssen_lexical_full_text",
        "vaccination_lexical_full_text",
        "covid_vaccination_lexical",
        "janssen_vaccination_lexical",
        "clean_v2_changed",
        "clean_v2_removed_chars",
        "model_text_chars",
    ]

    df[input_cols].to_csv(input_path, index=False, encoding="utf-8")
    df[metadata_cols].to_csv(metadata_path, index=False, encoding="utf-8")

    return {
        "name": name,
        "description": POPULATIONS[name],
        "input_csv": str(input_path),
        "metadata_csv": str(metadata_path),
        "sha256": {
            "input_csv": sha256_file(input_path),
            "metadata_csv": sha256_file(metadata_path),
        },
        **summarize_counts(df),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    docs = pd.read_parquet(args.input_parquet)
    docs["published_at"] = pd.to_datetime(docs["published_at"], errors="coerce")
    docs = docs.loc[docs["published_at"].dt.year == args.year].copy()
    docs = add_time_fields(docs)
    docs["model_text"] = docs["text_model"].fillna("").astype(str)
    docs["source_doc_id"] = docs["doc_id"].astype(str)
    docs["model_text_chars"] = docs["model_text"].str.len()

    flags = lexical_flags(docs["title"], docs["model_text"], max_chars=args.lexical_search_chars)
    full_text_flags = lexical_flags(docs["title"], docs["model_text"], max_chars=0)
    for col in flags.columns:
        docs[col] = flags[col]
    docs["janssen_lexical_full_text"] = full_text_flags["janssen_lexical"]
    docs["vaccination_lexical_full_text"] = full_text_flags["vaccination_lexical"]
    docs["covid_vaccination_lexical"] = docs["covid_lexical"] & docs["vaccination_lexical"]
    docs["janssen_vaccination_lexical"] = docs["janssen_lexical_full_text"] & docs["vaccination_lexical_full_text"]

    frames = population_frames(docs)
    populations = [write_population(name, frame, args.output_dir) for name, frame in frames.items()]

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "local_population_preparation_only",
        "cloud_inference_launched": False,
        "stop_rule": "Stop for user approval before Hugging Face inference.",
        "source_parquet": str(args.input_parquet),
        "source_parquet_sha256": sha256_file(args.input_parquet),
        "source_clean_v2_rows": int(len(pd.read_parquet(args.input_parquet, columns=["doc_id"]))),
        "year": int(args.year),
        "model_input_policy": "title plus clean-v2 body only; no topic/TEME metadata; input CSV exposes doc_id, source_doc_id and model_text only.",
        "lexical_definitions": {
            "covid_regex": COVID_REGEX,
            "vaccination_regex": VACCINATION_REGEX,
            "janssen_regex": JANSSEN_REGEX,
            "search_scope": (
                "title plus leading clean-v2 model_text characters"
                if args.lexical_search_chars and args.lexical_search_chars > 0
                else "title plus full clean-v2 model_text"
            ),
            "lexical_search_chars": int(args.lexical_search_chars),
            "covid_vaccination_candidate": "covid_regex AND vaccination_regex",
            "janssen_vaccination_context": "janssen_regex AND vaccination_regex",
        },
        "populations": populations,
        "locked_pipeline_for_future_approved_inference": {
            "system_id": "G0_E2",
            "gate": "G0 base/zero-shot Qwen/Qwen3-32B broad relevance gate",
            "gate_prompt": "configs/apertus_relevance_gate_prompt_v2_broad_vaccination.md",
            "extractor": "E2 Stage B adapted Qwen forced-relevant extractor",
            "extractor_adapter": "Pucciami/qwen3-32b-5c-media-lora-clean-v2-stage-b-al159-replay90-v1",
            "extractor_prompt": "configs/apertus_5c_sft_system_prompt_v7_stage2_forced_relevant.txt",
        },
    }
    manifest_path = args.report_dir / "prepare_2021_inference_population_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Wrote population manifest: {manifest_path}")
    for item in populations:
        print(f"{item['name']}: {item['n_rows']} rows")


if __name__ == "__main__":
    main()
