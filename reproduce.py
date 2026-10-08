"""Recompute the published numbers from the deposited files and check them.

Run from the repository root:

    python reproduce.py

Runs the locked-set confidence intervals, the event-window standardisation and the event-window
audit scoring into `out/`, then compares the checked values with `expected/headline_numbers.json`
(the values printed in the article and its supplement). Exits with status 1 if any value differs by
more than the tolerance. Needs only `pandas` and `numpy` (see `requirements.txt`); takes a minute or
two because of the 10,000-resample bootstraps.

What it recomputes from `data/`: every locked-set performance estimate and interval (Tables 1-2,
Supplementary Tables S1, S3 and S4, the paired differences between systems), the relevance routing
counts, the event-window z-scores, bands, p-values and prediction limits (Supplementary Table S6) with
the out-of-window alarm count, and the event-window audit proportions (overall and per event week).
What it cannot recompute is listed in README.md.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
TOLERANCE = 5e-4 + 1e-9  # the article prints three decimals; the 1e-9 admits exact halves such as 0.4375


def run(*args: str) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find(rows: list[dict[str, str]], **where: str) -> dict[str, str]:
    matches = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one row for {where}, found {len(matches)}")
    return matches[0]


def alarm_rows(event_window: list[dict[str, str]]) -> list[dict[str, str]]:
    """Out-of-window indicator-weeks (neither baseline, event window nor context week): one row per
    indicator-week flagging whether it exceeded the 95% limit, plus a summary row with the counts."""
    oow = [r for r in event_window
           if r["is_baseline"] == "False" and r["is_signal_window"] == "False" and r["is_context_week"] == "False"]
    rows = [{"period": r["period"], "indicator_id": r["indicator_id"],
             "above_95": str(int(r["band_recomputed"] != "within_reference"))} for r in oow]
    rows.append({"period": "all", "indicator_id": "all", "n_out_of_window": str(len(oow)),
                 "n_above_95": str(sum(int(r["above_95"]) for r in rows))})
    return rows


def routing_rows() -> list[dict[str, str]]:
    """How many of the 100 evaluation-set articles each relevance classifier routed to the extractor, and,
    among those the reference does not hold relevant, how many then received construct labels."""
    systems = {"G0_E2": "data/locked_set_comparisons/G0_E2/G0_E2_comparison.csv",
               "G0_E2_EN": "data/cross_lingual/G0_E2_EN/G0_E2_EN_comparison.csv",
               "A0_F2": "data/locked_set_comparisons/A0_F2/A0_F2_comparison.csv"}
    rows = []
    for system_id, path in systems.items():
        data = read_rows(ROOT / path)
        routed = [r for r in data if r["pred_relevance"] == "relevant"]
        false_pass = [r for r in routed if r["gold_relevance"] != "relevant"]
        labelled = [r for r in false_pass if r["pred_c5_presence_d1"] not in ("", "[]")]
        rows.append({"system_id": system_id, "n_routed_relevant": str(len(routed)),
                     "n_false_pass": str(len(false_pass)), "n_false_pass_with_constructs": str(len(labelled))})
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    run("scripts/compute_locked_performance_cis.py", "--output-dir", str(OUT))
    run("scripts/recompute_event_window_zscores.py", "--output-dir", str(OUT / "event_window"))
    run("scripts/score_event_window_audit.py", "data/event_window_audit/event_window_audit_export.csv",
        "--output-csv", str(OUT / "event_window_audit" / "audit_scores.csv"))
    run("scripts/compute_annotator_agreement.py", "--output-dir", str(OUT / "annotator_agreement"))
    run("scripts/compute_locked_performance_cis.py", "--matrix-dir", "data/cross_lingual",
        "--paired-comparison",
        "EN=data/cross_lingual/G0_E2_EN/G0_E2_EN_comparison.csv"
        "::SL=data/locked_set_comparisons/G0_E2/G0_E2_comparison.csv",
        "--output-dir", str(OUT / "cross_lingual"))
    greedy = "data/locked_set_comparisons/A0_F0/A0_F0_comparison.csv"
    run("scripts/compute_locked_performance_cis.py", "--matrix-dir", "data/decoding_check",
        "--paired-comparison",
        f"S20260417=data/decoding_check/A0_F0_S20260417/A0_F0_S20260417_comparison.csv::GREEDY={greedy}",
        "--paired-comparison", f"S7=data/decoding_check/A0_F0_S7/A0_F0_S7_comparison.csv::GREEDY={greedy}",
        "--output-dir", str(OUT / "decoding_check"))

    sources = {
        "f1": read_rows(OUT / "gold100_f1_metric_bootstrap_cis.csv"),
        "wilson": read_rows(OUT / "gold100_binomial_metric_wilson_cis.csv"),
        "paired": read_rows(OUT / "gold100_paired_metric_differences_bootstrap_cis.csv"),
        "valence": read_rows(OUT / "gold100_valence_metric_cis.csv"),
        "event_window": read_rows(OUT / "event_window" / "event_window_zscores_recomputed.csv"),
        "event_limits": read_rows(OUT / "event_window" / "event_window_limits.csv"),
        "agreement": read_rows(OUT / "annotator_agreement" / "annotator_agreement_with_cis.csv"),
        "f1_en": read_rows(OUT / "cross_lingual" / "gold100_f1_metric_bootstrap_cis.csv"),
        "wilson_en": read_rows(OUT / "cross_lingual" / "gold100_binomial_metric_wilson_cis.csv"),
        "paired_en": read_rows(OUT / "cross_lingual" / "gold100_paired_metric_differences_bootstrap_cis.csv"),
        "f1_decoding": read_rows(OUT / "decoding_check" / "gold100_f1_metric_bootstrap_cis.csv"),
        "audit": read_rows(OUT / "event_window_audit" / "audit_scores.csv"),
    }
    sources["alarms"] = alarm_rows(sources["event_window"])
    sources["routing"] = routing_rows()
    expected = json.loads((ROOT / "expected" / "headline_numbers.json").read_text(encoding="utf-8"))

    failures = 0
    print("\nCheck against the values printed in the article")
    for item in expected["checks"]:
        row = find(sources[item["source"]], **item["where"])
        for column, value in item["values"].items():
            got = float(row[column])
            ok = abs(got - value) <= item.get("tolerance", TOLERANCE)
            failures += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {item['label']:66s} {column:16s} {got:9.4f}  (article {value})")

    if failures:
        raise SystemExit(f"\n{failures} value(s) differ from the article.")
    print("\nAll checked values match the article.")


if __name__ == "__main__":
    main()
