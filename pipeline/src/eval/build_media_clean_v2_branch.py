import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.media_text import build_model_text


DEFAULT_DOCS = ROOT / "data" / "interim" / "documents_media.parquet"
DEFAULT_GOLD100 = ROOT / "data" / "processed" / "gold100_test_adjudicated.csv"
DEFAULT_GOLD300 = ROOT / "data" / "locked" / "gold300_train" / "gold300_train_adjudicated.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "clean_v2"
DEFAULT_REPORT_DIR = ROOT / "reports" / "data_cleaning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conservative clean_v2 artifact branch for obvious OCR/layout garbage."
    )
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    parser.add_argument("--gold100", default=str(DEFAULT_GOLD100))
    parser.add_argument("--gold300", default=str(DEFAULT_GOLD300))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_diff_df(df: pd.DataFrame, title_col: str, text_col: str) -> pd.DataFrame:
    out = df.copy()
    out["model_text_v1"] = out.apply(
        lambda row: build_model_text(row.get(title_col, ""), row.get(text_col, "")),
        axis=1,
    )
    out["model_text_v2"] = out.apply(
        lambda row: build_model_text(
            row.get(title_col, ""),
            row.get(text_col, ""),
            clean_obvious_artifacts=True,
        ),
        axis=1,
    )
    out["clean_v2_changed"] = out["model_text_v1"] != out["model_text_v2"]
    out["clean_v2_removed_chars"] = (
        out["model_text_v1"].str.len() - out["model_text_v2"].str.len()
    ).clip(lower=0)
    return out


def summarize_changes(df: pd.DataFrame, dataset_name: str) -> dict:
    changed = df[df["clean_v2_changed"]].copy()
    removed = df["clean_v2_removed_chars"]
    return {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_changed_rows": int(len(changed)),
        "share_changed_rows": float(len(changed) / len(df)) if len(df) else 0.0,
        "total_removed_chars": int(changed["clean_v2_removed_chars"].sum()),
        "median_removed_chars_changed": float(changed["clean_v2_removed_chars"].median()) if len(changed) else 0.0,
        "max_removed_chars_changed": int(changed["clean_v2_removed_chars"].max()) if len(changed) else 0,
        "n_changed_ge_10": int((removed >= 10).sum()),
        "n_changed_ge_20": int((removed >= 20).sum()),
        "n_changed_ge_50": int((removed >= 50).sum()),
        "n_changed_ge_100": int((removed >= 100).sum()),
    }


def sample_changed_rows(df: pd.DataFrame, include_title: bool = True, n: int = 10) -> pd.DataFrame:
    changed = df[df["clean_v2_changed"]].copy()
    if changed.empty:
        cols = ["doc_id", "old_preview", "new_preview", "clean_v2_removed_chars"]
        if include_title and "title" in df.columns:
            cols.insert(1, "title")
        return pd.DataFrame(columns=cols)

    cols = ["doc_id", "clean_v2_removed_chars"]
    if include_title and "title" in changed.columns:
        cols.insert(1, "title")
    sampled = changed.sort_values(
        ["clean_v2_removed_chars", "doc_id"], ascending=[False, True]
    ).head(n)[cols].copy()
    sampled["old_preview"] = (
        changed.sort_values(["clean_v2_removed_chars", "doc_id"], ascending=[False, True])
        .head(n)["model_text_v1"]
        .str.slice(0, 280)
        .str.replace("\n", " ", regex=False)
    )
    sampled["new_preview"] = (
        changed.sort_values(["clean_v2_removed_chars", "doc_id"], ascending=[False, True])
        .head(n)["model_text_v2"]
        .str.slice(0, 280)
        .str.replace("\n", " ", regex=False)
    )
    return sampled


def all_changed_rows(df: pd.DataFrame, include_title: bool = True) -> pd.DataFrame:
    changed = df[df["clean_v2_changed"]].copy()
    cols = ["doc_id", "clean_v2_removed_chars"]
    if include_title and "title" in changed.columns:
        cols.insert(1, "title")
    out = changed[cols].sort_values(
        ["clean_v2_removed_chars", "doc_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return out


def write_markdown_report(
    path: Path,
    docs_summary: dict,
    gold100_summary: dict,
    gold300_summary: dict,
    gold100_sample: pd.DataFrame,
    gold300_sample: pd.DataFrame,
    docs_sample: pd.DataFrame,
) -> None:
    lines = [
        "# Media Text Clean V2 Audit",
        "",
        "This branch applies a conservative, opt-in cleanup for obvious OCR/layout artifacts.",
        "It does not replace the current locked benchmark by default.",
        "",
        "## Change Counts",
        "",
        "| Dataset | Rows | Changed rows | Share | Total removed chars |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in [docs_summary, gold100_summary, gold300_summary]:
        lines.append(
            f"| {summary['dataset']} | {summary['n_rows']} | {summary['n_changed_rows']} | "
            f"{summary['share_changed_rows']:.3%} | {summary['total_removed_chars']} |"
        )

    lines.extend([
        "",
        "## Severity Bins",
        "",
        "| Dataset | >=10 chars removed | >=20 chars removed | >=50 chars removed | >=100 chars removed |",
        "|---|---:|---:|---:|---:|",
    ])
    for summary in [docs_summary, gold100_summary, gold300_summary]:
        lines.append(
            f"| {summary['dataset']} | {summary['n_changed_ge_10']} | {summary['n_changed_ge_20']} | "
            f"{summary['n_changed_ge_50']} | {summary['n_changed_ge_100']} |"
        )

    def add_sample_table(title: str, sample_df: pd.DataFrame) -> None:
        lines.extend(["", f"## {title}", ""])
        if sample_df.empty:
            lines.append("No changed rows detected.")
            return
        lines.append("| doc_id | removed chars | old preview | new preview |")
        lines.append("|---|---:|---|---|")
        for row in sample_df.to_dict(orient="records"):
            old_preview = str(row["old_preview"]).replace("|", "\\|")
            new_preview = str(row["new_preview"]).replace("|", "\\|")
            lines.append(
                f"| {row['doc_id']} | {row['clean_v2_removed_chars']} | {old_preview} | {new_preview} |"
            )

    add_sample_table("Corpus Sample", docs_sample)
    add_sample_table("Gold100 Sample", gold100_sample)
    add_sample_table("Gold300 Sample", gold300_sample)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    docs_df = pd.read_parquet(args.docs)
    gold100_df = pd.read_csv(args.gold100, dtype=str, keep_default_na=False)
    gold300_df = pd.read_csv(args.gold300, dtype=str, keep_default_na=False)

    docs_diff = build_diff_df(docs_df, "title", "text")
    gold100_diff = build_diff_df(gold100_df, "title", "text")
    gold300_diff = build_diff_df(gold300_df, "title", "text")

    docs_summary = summarize_changes(docs_diff, "documents_media")
    gold100_summary = summarize_changes(gold100_diff, "gold100_processed")
    gold300_summary = summarize_changes(gold300_diff, "gold300_locked_train")

    docs_sample = sample_changed_rows(docs_diff)
    gold100_sample = sample_changed_rows(gold100_diff)
    gold300_sample = sample_changed_rows(gold300_diff)
    gold100_all = all_changed_rows(gold100_diff)
    gold300_all = all_changed_rows(gold300_diff)

    docs_changed_csv = output_dir / "documents_media_clean_v2_changed_rows.csv"
    gold100_changed_csv = output_dir / "gold100_clean_v2_changed_rows.csv"
    gold300_changed_csv = output_dir / "gold300_clean_v2_changed_rows.csv"
    docs_sample.to_csv(docs_changed_csv, index=False, encoding="utf-8")
    gold100_all.to_csv(gold100_changed_csv, index=False, encoding="utf-8")
    gold300_all.to_csv(gold300_changed_csv, index=False, encoding="utf-8")

    docs_clean = docs_df.copy()
    docs_clean["text_model"] = docs_diff["model_text_v2"]
    docs_clean["clean_v2_changed"] = docs_diff["clean_v2_changed"]
    docs_clean["clean_v2_removed_chars"] = docs_diff["clean_v2_removed_chars"]
    docs_clean_parquet = output_dir / "documents_media_clean_v2.parquet"
    docs_clean.to_parquet(docs_clean_parquet, index=False)

    gold100_clean = gold100_diff[["doc_id", "labelstudio_task_id", "source_doc_id"]].copy()
    gold100_clean["text"] = gold100_diff["model_text_v2"]
    gold100_clean_csv = output_dir / "gold100_test_apertus_input_clean_v2.csv"
    gold100_clean.to_csv(gold100_clean_csv, index=False, encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "docs_input": str(Path(args.docs)),
        "gold100_input": str(Path(args.gold100)),
        "gold300_input": str(Path(args.gold300)),
        "outputs": {
            "docs_clean_parquet": str(docs_clean_parquet),
            "gold100_clean_csv": str(gold100_clean_csv),
            "docs_changed_csv": str(docs_changed_csv),
            "gold100_changed_csv": str(gold100_changed_csv),
            "gold300_changed_csv": str(gold300_changed_csv),
        },
        "summaries": [docs_summary, gold100_summary, gold300_summary],
        "sha256": {
            "docs_clean_parquet": sha256_file(docs_clean_parquet),
            "gold100_clean_csv": sha256_file(gold100_clean_csv),
            "docs_changed_csv": sha256_file(docs_changed_csv),
            "gold100_changed_csv": sha256_file(gold100_changed_csv),
            "gold300_changed_csv": sha256_file(gold300_changed_csv),
        },
        "policy": {
            "branch_name": "clean_v2",
            "default_benchmark_unchanged": True,
            "cleaner_mode": "conservative_opt_in",
        },
    }

    manifest_path = report_dir / "media_text_clean_v2_audit_2026-05-07.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    markdown_path = report_dir / "media_text_clean_v2_audit_2026-05-07.md"
    write_markdown_report(
        markdown_path,
        docs_summary,
        gold100_summary,
        gold300_summary,
        gold100_sample,
        gold300_sample,
        docs_sample,
    )

    print(f"Saved report JSON: {manifest_path}")
    print(f"Saved report MD: {markdown_path}")
    print(f"Saved docs clean parquet: {docs_clean_parquet}")
    print(f"Saved gold100 clean CSV: {gold100_clean_csv}")
    print(json.dumps({"docs": docs_summary, "gold100": gold100_summary, "gold300": gold300_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
